import os
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from internvl_temporal_adapter import TemporalVideoEncoder
from gavd_skeleton_dataset import (
    GavdSkeletonDataset,
    collect_labeled_sequences,
    video_level_train_test_split,
    TOP7_LABELS,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# For InternVL 1B, limit to 32 image frames per sequence
WINDOW_SIZE = 32
BATCH_SIZE = 1  # images + skeletons are heavier, start with 1
EPOCHS = 20
LR = 5e-4

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GAVD-sequences")
CKPT_PATH = "best_skeleton_image_classifier.pt"


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


def collate_fn(batch, device):
    """
    `batch` is a dict:
      {
        'skeleton': FloatTensor [B, W, 46],
        'images':   FloatTensor [B, W, 3, H, W],
        'label':    LongTensor [B]
      }
    """
    skeletons = batch["skeleton"].to(device)
    images = batch["images"].to(device)
    labels = batch["label"].to(device)

    return skeletons, images, labels


def evaluate(video_encoder, classifier, data_loader, device):
    video_encoder.eval()
    classifier.eval()
    correct = 0
    total = 0

    num_classes = len(TOP7_LABELS)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            skeletons, images, labels = collate_fn(batch, device)

            feats = video_encoder(
                pixel_values=images,
                skeleton_feats=skeletons,
            )
            feats = feats.float()

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

    # EVL-style temporal encoder on top of frozen InternVL vision backbone
    video_encoder = TemporalVideoEncoder(
        base_model,
        max_frames=WINDOW_SIZE,
        use_skeleton=True,
    ).to(DEVICE)
    video_encoder.eval()

    hidden_size = video_encoder.hidden_size
    num_classes = len(TOP7_LABELS)

    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    train_loader, test_loader, train_samples = build_dataloaders()
    class_weights = compute_class_weights(train_samples).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(video_encoder.parameters()) + list(classifier.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    # InternVL backbone is frozen inside TemporalVideoEncoder

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_macro_f1 = -1.0
    start_epoch = 1

    # Optional resume from best checkpoint
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        try:
            video_encoder.load_state_dict(ckpt["video_encoder"])
            classifier.load_state_dict(ckpt["classifier"])
            best_macro_f1 = ckpt.get("macro_f1", -1.0)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(
                f"Resuming from {CKPT_PATH}: "
                f"epoch={ckpt.get('epoch')}, best_macro_f1={best_macro_f1*100:.2f}%"
            )
        except KeyError:
            print(f"Found {CKPT_PATH} but missing expected keys; starting from scratch.")

    for epoch in range(start_epoch, EPOCHS + 1):
        video_encoder.train()
        classifier.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - train", leave=False):
            skeletons, images, labels = collate_fn(batch, DEVICE)

            optimizer.zero_grad()

            feats = video_encoder(
                pixel_values=images,
                skeleton_feats=skeletons,
            )
            feats = feats.float()

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

        _, macro_f1 = evaluate(video_encoder, classifier, test_loader, DEVICE)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "video_encoder": video_encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "macro_f1": best_macro_f1,
                    "epoch": epoch,
                },
                "best_skeleton_image_classifier.pt",
            )
            print(f"Saved new best image+skeleton model (macro-F1={macro_f1*100:.2f}%)")


if __name__ == "__main__":
    main()


