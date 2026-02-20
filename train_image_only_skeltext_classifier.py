import os
import json
import argparse
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from internvl_skeleton_adapter import InternVLWithSkeleton
from gavd_skeleton_dataset import (
    GavdSkeletonDataset,
    collect_labeled_sequences,
    video_level_train_test_split,
    TOP7_LABELS,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 32
BATCH_SIZE = 1
EPOCHS = 20
LR = 5e-4

MAX_SKELETON_TEXT_FRAMES = 32
MAX_TEXT_TOKENS = 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "GAVD-sequences")
HSMR_TEXT_DIR = os.path.join(BASE_DIR, "GAVD-HSMR-text")

CKPT_PATH = "best_image_only_skeltext_classifier.pt"
STATE_PATH = "image_only_skeltext_train_state.pt"

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
        train_samples, test_samples = video_level_train_test_split(samples, train_ratio=0.8)
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
        video_dir=VIDEO_DIR if not ARGS.splits_dir else BASE_DIR,
    )
    test_ds = GavdSkeletonDataset(
        test_samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=VIDEO_DIR if not ARGS.splits_dir else BASE_DIR,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_samples, labels


def compute_class_weights(train_samples: List[Dict], labels: List[str]) -> torch.Tensor:
    counts = [0] * len(labels)
    for s in train_samples:
        counts[s["label_idx"]] += 1

    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    counts_tensor = torch.clamp(counts_tensor, min=1.0)
    N = counts_tensor.sum()
    K = float(len(labels))
    weights = N / (K * counts_tensor)
    weights = weights / weights.mean()
    print("\nClass weights:", weights.tolist())
    return weights


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


def evaluate(model, classifier, data_loader, tokenizer, device):
    model.eval()
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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=images,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]  # [B, T_img+L, D]

            T_img = WINDOW_SIZE
            img_hidden = hidden[:, :T_img, :]
            feats = img_hidden.mean(dim=1).float()

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
        help="If set, loads combined_train/test.csv from this directory to train on GAVD+DCM.",
    )
    ARGS = parser.parse_args()
    if ARGS.splits_dir:
        ARGS.splits_dir = _resolve(ARGS.splits_dir)
        if not os.path.isdir(ARGS.splits_dir):
            raise FileNotFoundError(f"--splits-dir not found: {ARGS.splits_dir}")

    tokenizer, base_model, _ = load_model(device=DEVICE)
    img_model = InternVLWithSkeleton(base_model).to(DEVICE)
    img_model.eval()

    hidden_size = img_model.hidden_size
    train_loader, test_loader, train_samples, LABELS = build_dataloaders()
    num_classes = len(LABELS)
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    class_weights = compute_class_weights(train_samples, labels=LABELS).to(DEVICE)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_macro_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        img_model.eval()
        classifier.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - train", leave=False):
            images, labels, input_ids, attention_mask = collate_fn(batch, tokenizer, DEVICE)

            optimizer.zero_grad()

            with torch.no_grad():
                outputs = img_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=images,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[-1]
                T_img = WINDOW_SIZE
                img_hidden = hidden[:, :T_img, :]
                feats = img_hidden.mean(dim=1).float()

            logits = classifier(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        print(f"\nEpoch {epoch}/{EPOCHS} - train loss: {train_loss:.4f}, acc: {train_acc*100:.2f}%")

        _, macro_f1 = evaluate(img_model, classifier, test_loader, tokenizer, DEVICE)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "image_model": img_model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "macro_f1": best_macro_f1,
                    "epoch": epoch,
                },
                CKPT_PATH,
            )
            print(f"Saved new best image-only+skeltext model (macro-F1={macro_f1*100:.2f}%)")

        torch.save(
            {
                "image_model": img_model.state_dict(),
                "classifier": classifier.state_dict(),
                "macro_f1": macro_f1,
                "epoch": epoch,
            },
            STATE_PATH,
        )
        print(f"Saved last-epoch image-only+skeltext model to {STATE_PATH}")


if __name__ == "__main__":
    main()
