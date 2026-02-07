import os
import json
import argparse
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from internvl_temporal_adapter import FrameAlignedEVLDecoder
from gavd_skeleton_dataset import (
    GavdSkeletonDataset,
    collect_labeled_sequences,
    video_level_train_test_split,
    TOP7_LABELS,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 32
BATCH_SIZE = 1

MAX_SKELETON_TEXT_FRAMES = 16
MAX_TEXT_TOKENS = 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "GAVD-sequences")
HSMR_TEXT_DIR = os.path.join(BASE_DIR, "GAVD-HSMR-text")

STATE_PATH = "evl_framealigned_skeltext_train_state.pt"

ARGS = None
LABELS: List[str] = []


def _resolve(p: str) -> str:
    p = str(p)
    if not os.path.isabs(p):
        p = os.path.join(BASE_DIR, p)
    return os.path.normpath(p)


def _load_samples_from_split_csv(csv_path: str, label_to_idx: Dict[str, int]) -> List[Dict]:
    """
    Load samples from a split CSV produced by `make_gavd_plus_dcm_splits.py`.
    Expected columns (minimum): seq_id,label,skeleton_path,video_path
    Optional: text_path
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"seq_id", "label", "skeleton_path", "video_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV is missing columns {sorted(missing)}: {csv_path}")

    has_text = "text_path" in df.columns

    samples: List[Dict] = []
    for _, r in df.iterrows():
        seq_id = str(r["seq_id"])
        label_str = str(r["label"])
        if label_str not in label_to_idx:
            raise ValueError(f"Unknown label {label_str!r} in {csv_path}. Known: {sorted(label_to_idx.keys())}")
        meta = {
            "seq_id": seq_id,
            "path": _resolve(r["skeleton_path"]),
            "video_path": _resolve(r["video_path"]),
            "label_str": label_str,
            "label_idx": int(label_to_idx[label_str]),
        }
        if has_text and isinstance(r["text_path"], str) and r["text_path"].strip() != "":
            meta["text_path"] = _resolve(r["text_path"])
        samples.append(meta)
    return samples


def build_test_loader() -> DataLoader:
    assert ARGS is not None

    if ARGS.splits_dir:
        test_csv = os.path.join(ARGS.splits_dir, "combined_test.csv")
        labels = TOP7_LABELS + ["dcm"]
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        test_samples = _load_samples_from_split_csv(test_csv, label_to_idx=label_to_idx)
    else:
        test_samples = None

    if test_samples is None:
        samples = collect_labeled_sequences()
        _, test_samples = video_level_train_test_split(samples, train_ratio=0.8, seed=42)
        # Default GAVD skeleton-text path
        for s in test_samples:
            s["text_path"] = os.path.join(HSMR_TEXT_DIR, f"HSMR-{s['seq_id']}.jsonl")

    test_ds = GavdSkeletonDataset(
        test_samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=VIDEO_DIR,
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return test_loader


def _load_skeleton_text(text_path: str, start: int, window_size: int) -> str:
    if not isinstance(text_path, str) or text_path.strip() == "" or not os.path.exists(text_path):
        return ""

    records = []
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(obj)
    except OSError:
        return ""

    if not records:
        return ""

    max_idx = len(records) - 1
    num_available = min(window_size, len(records) - start if start <= max_idx else 0)
    if num_available <= 0:
        num_available = min(len(records), window_size)

    num_lines = min(num_available, MAX_SKELETON_TEXT_FRAMES)
    if num_lines <= 0:
        return ""

    step = max(1, window_size // num_lines)
    lines: List[str] = []
    used = 0
    frame_idx = start
    while used < num_lines and frame_idx <= start + window_size - 1:
        idx = min(frame_idx, max_idx)
        rec = records[idx]
        skel_str = rec.get("skel", "")
        frame_no = rec.get("frame", idx)
        if skel_str:
            lines.append(f"Frame {frame_no}: {skel_str}")
            used += 1
        frame_idx += step

    return "\n".join(lines)


def collate_fn(batch, tokenizer, device):
    images = batch["images"].to(device)
    labels = batch["label"].to(device)
    seq_ids = batch["seq_id"]
    starts = batch["start"]
    text_paths = batch.get("text_path", [""] * images.size(0))

    base_prompt = (
        "You are an expert gait clinician. Based on the available gait information, "
        "classify the patient's gait pattern.\n\n"
        "Gait pattern definitions:\n"
        "- abnormal: any gait pattern that deviates from normal but does not fit the specific patterns below.\n"
        "- myopathic: waddling or Trendelenburg-type gait due to proximal muscle weakness.\n"
        "- exercise: exaggerated, energetic, or performance-like gait related to sport or exercise.\n"
        "- normal: typical, symmetric gait without obvious abnormalities.\n"
        "- style: exaggerated or stylistic walking pattern without clear neurological or orthopedic cause.\n"
        "- cerebral palsy: spastic, scissoring, toe-walking, or crouched gait typical of cerebral palsy.\n"
        "- parkinsons: shuffling, stooped posture, reduced arm swing, and festination typical of Parkinson's disease.\n\n"
        "Below are per-frame skeleton parameters extracted from the same gait sequence.\n"
        "Use both the visual gait information and these skeleton parameters to internally decide which class is most likely. "
        "You do not need to output the class name."
    )

    prompts: List[str] = []
    B = images.size(0)
    for i in range(B):
        start = int(starts[i].item())
        text_path = text_paths[i] if isinstance(text_paths, list) else ""
        skel_text = _load_skeleton_text(str(text_path), start, WINDOW_SIZE)
        if skel_text:
            full_prompt = base_prompt + "\n\nSkeleton parameters:\n" + skel_text
        else:
            full_prompt = base_prompt + "\n\n(No skeleton parameters available for this sequence.)"
        prompts.append(full_prompt)

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    return images, labels, input_ids, attention_mask


def evaluate(decoder, language_model, classifier, data_loader, tokenizer, device):
    decoder.eval()
    classifier.eval()
    correct = 0
    total = 0

    num_classes = len(LABELS)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            images, labels, input_ids, attention_mask = collate_fn(batch, tokenizer, device)

            frame_tokens = decoder(pixel_values=images)  # [B, T, D]
            text_embeds = language_model.get_input_embeddings()(input_ids)  # [B, L, D]
            frame_tokens = frame_tokens.to(dtype=text_embeds.dtype)

            B, T, _ = frame_tokens.shape
            video_mask = torch.ones(B, T, dtype=attention_mask.dtype, device=device)

            inputs_embeds = torch.cat([frame_tokens, text_embeds], dim=1)
            fused_mask = torch.cat([video_mask, attention_mask], dim=1)

            outputs = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=fused_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            video_hidden = hidden[:, :T, :]
            feats = video_hidden.mean(dim=1).float()

            logits = classifier(feats)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for c in range(num_classes):
                tp[c] += ((preds == c) & (labels == c)).sum().item()
                fp[c] += ((preds == c) & (labels != c)).sum().item()
                fn[c] += ((preds != c) & (labels == c)).sum().item()

    acc = correct / max(1, total)

    per_class_f1 = []
    for c in range(num_classes):
        precision = tp[c] / max(1, tp[c] + fp[c])
        recall = tp[c] / max(1, tp[c] + fn[c])
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class_f1.append(f1)

    macro_f1 = sum(per_class_f1) / num_classes

    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"Test macro-F1: {macro_f1 * 100:.2f}%")
    print("Per-class F1:")
    for idx, f1 in enumerate(per_class_f1):
        print(f"  {LABELS[idx]}: {f1 * 100:.2f}%")

    return acc, macro_f1


def main():
    global ARGS, LABELS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="",
        help="If set, evaluates on combined_test.csv in this directory (supports GAVD+DCM).",
    )
    ARGS = parser.parse_args()
    if ARGS.splits_dir:
        ARGS.splits_dir = _resolve(ARGS.splits_dir)
        if not os.path.isdir(ARGS.splits_dir):
            raise FileNotFoundError(f"--splits-dir not found: {ARGS.splits_dir}")

    if not os.path.exists(STATE_PATH):
        raise FileNotFoundError(
            f"{STATE_PATH} not found. Run train_evl_framealigned_skeltext_classifier.py first."
        )

    tokenizer, base_model, _ = load_model(device=DEVICE)
    language_model = getattr(base_model, "language_model", base_model)
    language_model.requires_grad_(False)

    decoder = FrameAlignedEVLDecoder(
        base_model,
        max_frames=WINDOW_SIZE,
        num_layers=3,
        num_heads=4,
    ).to(DEVICE)

    hidden_size = decoder.hidden_size
    # Infer classifier output size from training state (so eval matches the trained model).
    num_classes = len(TOP7_LABELS)
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    state = torch.load(STATE_PATH, map_location=DEVICE)
    decoder.load_state_dict(state["decoder"])
    classifier.load_state_dict(state["classifier"])
    print(
        f"Loaded training state from {STATE_PATH}: "
        f"epoch={state.get('epoch')}, best_macro_f1={state.get('best_macro_f1', 0.0)*100:.2f}%"
    )

    trained_num_classes = int(classifier.weight.shape[0])
    if trained_num_classes == len(TOP7_LABELS):
        LABELS = TOP7_LABELS
    elif trained_num_classes == len(TOP7_LABELS) + 1:
        LABELS = TOP7_LABELS + ["dcm"]
    else:
        raise ValueError(
            f"Unexpected classifier output size {trained_num_classes}. "
            f"Expected {len(TOP7_LABELS)} (GAVD-only) or {len(TOP7_LABELS)+1} (GAVD+DCM)."
        )

    if ARGS.splits_dir and "dcm" in (set(TOP7_LABELS + ["dcm"]) - set(LABELS)):
        raise ValueError(
            "You passed --splits-dir (combined eval) but the loaded model appears to be GAVD-only (7 classes). "
            "Either retrain the skeltext model with DCM included (8 classes) or evaluate without --splits-dir."
        )

    test_loader = build_test_loader()
    evaluate(decoder, language_model, classifier, test_loader, tokenizer, DEVICE)


if __name__ == "__main__":
    main()

