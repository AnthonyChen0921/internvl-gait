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

MAX_SKELETON_TEXT_FRAMES = 32
MAX_TEXT_TOKENS = 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "GAVD-sequences")
HSMR_TEXT_DIR = os.path.join(BASE_DIR, "GAVD-HSMR-text")

CKPT_PATH = "best_evl_framealigned_skeltext_classifier.pt"
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
    Expected columns: seq_id,label,skeleton_path,video_path,text_path
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


def _print_label_distribution(samples: List[Dict], labels: List[str], title: str) -> None:
    counts = [0] * len(labels)
    for s in samples:
        idx = int(s["label_idx"])
        if 0 <= idx < len(counts):
            counts[idx] += 1
    parts = [f"{labels[i]}: {counts[i]}" for i in range(len(labels))]
    print(f"\n{title}: " + " ".join(parts))


def build_dataloaders() -> Tuple[DataLoader, DataLoader, List[Dict], List[str]]:
    """
    Two modes:
      - Default (GAVD-only): uses collect_labeled_sequences() + video-level split.
      - Combined (GAVD+DCM): pass --splits-dir to load combined_train/test.csv (leak-free).
    """
    assert ARGS is not None

    if ARGS.splits_dir:
        splits_dir = ARGS.splits_dir
        train_csv = os.path.join(splits_dir, "combined_train.csv")
        test_csv = os.path.join(splits_dir, "combined_test.csv")

        labels = TOP7_LABELS + ["dcm"]  # single-label DCM splits
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        train_samples = _load_samples_from_split_csv(train_csv, label_to_idx=label_to_idx)
        test_samples = _load_samples_from_split_csv(test_csv, label_to_idx=label_to_idx)
    else:
        labels = TOP7_LABELS
        samples = collect_labeled_sequences()
        train_samples, test_samples = video_level_train_test_split(samples, train_ratio=0.8, seed=42)
        # Default skeleton-text path for GAVD
        for s in train_samples:
            s["text_path"] = os.path.join(HSMR_TEXT_DIR, f"HSMR-{s['seq_id']}.jsonl")
        for s in test_samples:
            s["text_path"] = os.path.join(HSMR_TEXT_DIR, f"HSMR-{s['seq_id']}.jsonl")

    print(f"\nTrain sequences: {len(train_samples)}")
    print(f"Test sequences: {len(test_samples)}")
    _print_label_distribution(train_samples, labels, title="Train label distribution")
    _print_label_distribution(test_samples, labels, title="Test label distribution")
    if not ARGS.splits_dir:
        train_videos = {s["video_id"] for s in train_samples}
        test_videos = {s["video_id"] for s in test_samples}
        print(f"Train videos: {len(train_videos)}")
        print(f"Test videos: {len(test_videos)}")
        print(f"Overlap videos between train/test: {len(train_videos & test_videos)}")

    train_ds = GavdSkeletonDataset(
        train_samples,
        window_size=WINDOW_SIZE,
        train=True,
        with_images=True,
        video_dir=VIDEO_DIR if not ARGS.splits_dir else BASE_DIR,  # ignored when meta['video_path'] is present
    )
    test_ds = GavdSkeletonDataset(
        test_samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=VIDEO_DIR if not ARGS.splits_dir else BASE_DIR,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_samples, labels


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


def _build_classification_prompts(batch) -> List[str]:
    images = batch["images"]
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
        _ = seq_ids[i]
        start = int(starts[i].item())
        text_path = text_paths[i] if isinstance(text_paths, list) else ""
        skel_text = _load_skeleton_text(str(text_path), start, WINDOW_SIZE)
        if skel_text:
            full_prompt = base_prompt + "\n\nSkeleton parameters:\n" + skel_text
        else:
            full_prompt = base_prompt + "\n\n(No skeleton parameters available for this sequence.)"
        prompts.append(full_prompt)
    return prompts


def _build_report_prompt(label: str) -> str:
    return (
        "You are a clinical assistant. I have already analyzed the visual features and "
        f"identified the condition as {label}.\n"
        "Please generate a detailed clinical report that supports this diagnosis. "
        f"Focus on describing observations and evidence that explain why this is {label}."
    )


def _prepare_batch(batch, device):
    images = batch["images"].to(device)
    labels = batch["label"].to(device)
    seq_ids = batch["seq_id"]
    starts = batch["start"]
    return images, labels, seq_ids, starts


def _predict_labels(decoder, language_model, classifier, tokenizer, batch, device) -> torch.Tensor:
    images, labels, _, _ = _prepare_batch(batch, device)
    prompts = _build_classification_prompts(batch)

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    frame_tokens = decoder(pixel_values=images)  # [B, T, D]
    text_embeds = language_model.get_input_embeddings()(input_ids)  # [B, L, D]
    frame_tokens = frame_tokens.to(dtype=text_embeds.dtype)

    B, T, _ = frame_tokens.shape
    video_mask = torch.ones(B, T, dtype=attention_mask.dtype, device=device)

    inputs_embeds = torch.cat([frame_tokens, text_embeds], dim=1)  # [B, T+L, D]
    fused_mask = torch.cat([video_mask, attention_mask], dim=1)  # [B, T+L]

    outputs = language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=fused_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]  # [B, T+L, D]
    video_hidden = hidden[:, :T, :]
    feats = video_hidden.mean(dim=1).float()

    logits = classifier(feats)
    preds = logits.argmax(dim=-1)
    return preds


def _generate_reports(language_model, tokenizer, labels: List[str], preds: torch.Tensor) -> List[str]:
    prompts = [_build_report_prompt(labels[int(p.item())]) for p in preds]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    gen_kwargs = {
        "max_new_tokens": ARGS.max_new_tokens,
        "do_sample": ARGS.do_sample,
        "temperature": ARGS.temperature,
        "top_p": ARGS.top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    outputs = language_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    reports: List[str] = []
    for i, out_ids in enumerate(outputs):
        full_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        prompt = prompts[i]
        if full_text.startswith(prompt):
            report = full_text[len(prompt) :].strip()
        else:
            report = full_text.strip()
        reports.append(report)
    return reports


def main():
    global ARGS, LABELS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="",
        help="If set, loads combined_train/test.csv from this directory to evaluate on GAVD+DCM.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to run generation on.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to best checkpoint. Defaults to best_evl_framealigned_skeltext_classifier.pt",
    )
    parser.add_argument(
        "--use-last",
        action="store_true",
        help="Load from evl_framealigned_skeltext_train_state.pt instead of best checkpoint.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="evl_framealigned_skeltext_reports.jsonl",
        help="Output path for generated reports.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = no limit).")
    ARGS = parser.parse_args()

    if ARGS.splits_dir:
        ARGS.splits_dir = _resolve(ARGS.splits_dir)
        if not os.path.isdir(ARGS.splits_dir):
            raise FileNotFoundError(f"--splits-dir not found: {ARGS.splits_dir}")

    tokenizer, base_model, _ = load_model(device=DEVICE)
    language_model = getattr(base_model, "language_model", base_model)
    language_model.requires_grad_(False)
    language_model.eval()

    decoder = FrameAlignedEVLDecoder(
        base_model,
        max_frames=WINDOW_SIZE,
        num_layers=3,
        num_heads=4,
    ).to(DEVICE)
    decoder.eval()

    hidden_size = decoder.hidden_size
    train_loader, test_loader, _, LABELS = build_dataloaders()
    num_classes = len(LABELS)
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)
    classifier.eval()

    ckpt_path = _resolve(ARGS.checkpoint) if ARGS.checkpoint else _resolve(CKPT_PATH)
    state_path = _resolve(STATE_PATH)

    if ARGS.use_last:
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"State file not found: {state_path}")
        state = torch.load(state_path, map_location=DEVICE)
        decoder.load_state_dict(state["decoder"])
        classifier.load_state_dict(state["classifier"])
        print(f"Loaded last state from {state_path}")
    else:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        decoder.load_state_dict(ckpt["decoder"])
        classifier.load_state_dict(ckpt["classifier"])
        print(f"Loaded best checkpoint from {ckpt_path}")

    data_loader = train_loader if ARGS.split == "train" else test_loader
    output_path = _resolve(ARGS.output_jsonl)
    total_written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for batch in tqdm(data_loader, desc=f"Generating reports ({ARGS.split})"):
            preds = _predict_labels(decoder, language_model, classifier, tokenizer, batch, DEVICE)
            reports = _generate_reports(language_model, tokenizer, LABELS, preds)

            seq_ids = batch["seq_id"]
            labels = batch["label"]
            for i in range(len(reports)):
                seq_id = seq_ids[i]
                true_label = LABELS[int(labels[i].item())]
                pred_label = LABELS[int(preds[i].item())]
                record = {
                    "seq_id": seq_id,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "report": reports[i],
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                total_written += 1
                if ARGS.limit and total_written >= ARGS.limit:
                    break
            if ARGS.limit and total_written >= ARGS.limit:
                break

    print(f"\nWrote {total_written} reports to {output_path}")


if __name__ == "__main__":
    main()
