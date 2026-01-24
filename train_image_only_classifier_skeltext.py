import os
import json
from typing import Dict, List

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

# To control memory usage inside the language model, we (1) limit how many
# frames of skeleton text we include, and (2) cap the total number of text
# tokens passed to the tokenizer.
MAX_SKELETON_TEXT_FRAMES = 16
MAX_TEXT_TOKENS = 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "GAVD-sequences")
HSMR_TEXT_DIR = os.path.join(BASE_DIR, "GAVD-HSMR-text")

CKPT_PATH = "best_image_only_classifier_skeltext.pt"
STATE_PATH = "image_only_skeltext_train_state.pt"


def build_dataloaders():
    samples = collect_labeled_sequences()
    train_samples, test_samples = video_level_train_test_split(samples, train_ratio=0.8)

    # Video-level summary
    train_videos = {s["video_id"] for s in train_samples}
    test_videos = {s["video_id"] for s in test_samples}
    print(f"\nTrain sequences: {len(train_samples)}")
    print(f"Test sequences: {len(test_samples)}")
    print(f"Train videos: {len(train_videos)}")
    print(f"Test videos: {len(test_videos)}")
    print(f"Overlap videos between train/test: {len(train_videos & test_videos)}")

    train_ds = GavdSkeletonDataset(
        train_samples,
        window_size=WINDOW_SIZE,
        train=True,
        with_images=True,
        video_dir=VIDEO_DIR,
    )
    test_ds = GavdSkeletonDataset(
        test_samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=VIDEO_DIR,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_samples


def compute_class_weights(train_samples: List[Dict]) -> torch.Tensor:
    counts = [0] * len(TOP7_LABELS)
    for s in train_samples:
        counts[s["label_idx"]] += 1

    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    counts_tensor = torch.clamp(counts_tensor, min=1.0)
    N = counts_tensor.sum()
    K = float(len(TOP7_LABELS))
    weights = N / (K * counts_tensor)
    weights = weights / weights.mean()
    print("\nClass weights:", weights.tolist())
    return weights


def _load_skeleton_text(seq_id: str, start: int, window_size: int) -> str:
    """
    Load per-frame skeleton text from GAVD-HSMR-text/HSMR-{seq_id}.jsonl and
    align it with the same temporal window [start, start+window_size).
    """
    path = os.path.join(HSMR_TEXT_DIR, f"HSMR-{seq_id}.jsonl")
    if not os.path.exists(path):
        return ""

    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
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
    """
    `batch` is a dict:
      {
        "skeleton": FloatTensor [B, W, 46],  # unused here
        "images":   FloatTensor [B, W, 3, H, W],
        "label":    LongTensor [B],
        "seq_id":   list[str] length B,
        "start":    LongTensor [B]
      }
    """
    images = batch["images"].to(device)
    labels = batch["label"].to(device)
    seq_ids = batch["seq_id"]
    starts = batch["start"]

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
        seq_id = seq_ids[i]
        start = int(starts[i].item())
        skel_text = _load_skeleton_text(seq_id, start, WINDOW_SIZE)
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

    num_classes = len(TOP7_LABELS)
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
            img_hidden = hidden[:, :WINDOW_SIZE, :]
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
        print(f"  {TOP7_LABELS[idx]}: {f1 * 100:.2f}%")

    return acc, macro_f1


def main():
    tokenizer, base_model, _ = load_model(device=DEVICE)
    img_model = InternVLWithSkeleton(base_model).to(DEVICE)
    img_model.eval()

    hidden_size = img_model.hidden_size
    num_classes = len(TOP7_LABELS)
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    train_loader, test_loader, train_samples = build_dataloaders()
    class_weights = compute_class_weights(train_samples).to(DEVICE)

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
                img_hidden = hidden[:, :WINDOW_SIZE, :]
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

