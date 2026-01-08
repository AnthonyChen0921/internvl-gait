import os
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from gavd_skeleton_dataset import (
    GavdSkeletonDataset,
    collect_labeled_sequences,
    video_level_train_test_split,
    TOP7_LABELS,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use 32 frames per sequence for the 1B model
WINDOW_SIZE = 32
BATCH_SIZE = 1
EPOCHS = 20
LR = 5e-4
STATE_PATH = "video_token_train_state.pt"
BEST_PATH = "best_video_token_classifier.pt"

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GAVD-sequences")


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
        'skeleton': FloatTensor [B, W, 46],  # unused here
        'images':   FloatTensor [B, W, 3, H, W],
        'label':    LongTensor [B]
      }
    """
    images = batch["images"].to(device)
    labels = batch["label"].to(device)
    return images, labels


def extract_video_features(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Use InternVL's frozen vision backbone to obtain a single feature vector per video
    directly from video/image tokens (no text, no language model).

    Args:
        model: InternVL chat model with `.extract_feature`.
        pixel_values: [B, T, 3, H, W] tensor of frames.

    Returns:
        video_features: [B, D] tensor.
    """
    if pixel_values.ndim != 5:
        raise ValueError(f"pixel_values must be [B, T, 3, H, W], got {pixel_values.shape}")

    # Determine device and dtype from the frozen model
    try:
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype
    except StopIteration:
        device = pixel_values.device
        dtype = torch.float32

    B, T, C, H, W = pixel_values.shape

    # Flatten batch and time so we can call extract_feature on images
    flat = pixel_values.to(device=device, dtype=dtype).view(B * T, C, H, W)  # [B*T, 3, H, W]

    # Use InternVL's vision encoder to get tokens per frame
    with torch.no_grad():
        vit_tokens = model.extract_feature(flat)  # [B*T, N_tokens, D]

    # First pool spatial tokens per frame (mean over patches)
    frame_tokens = vit_tokens.mean(dim=1)  # [B*T, D]

    # Then pool over time to obtain a video-level feature
    video_tokens = frame_tokens.view(B, T, -1)  # [B, T, D]
    video_features = video_tokens.mean(dim=1)  # [B, D]

    return video_features.float()


def evaluate(model, classifier, data_loader, device):
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
            images, labels = collate_fn(batch, device)

            feats = extract_video_features(model, images)  # [B, D]
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
    # We only need the InternVL model here; tokenizer is unused.
    tokenizer, base_model, _ = load_model(device=DEVICE)  # noqa: F841

    # Freeze InternVL; we will only train the classifier on top of video tokens.
    base_model.eval()
    base_model.requires_grad_(False)

    # Hidden size is taken from the language model config if present
    language_model = getattr(base_model, "language_model", None)
    if language_model is not None:
        hidden_size = language_model.config.hidden_size
    else:
        hidden_size = base_model.config.hidden_size

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

    start_epoch = 1
    best_macro_f1 = -1.0

    # Simple resume logic from the last-epoch state if available
    if os.path.exists(STATE_PATH):
        try:
            state = torch.load(STATE_PATH, map_location=DEVICE)
            classifier.load_state_dict(state["classifier"])
            best_macro_f1 = state.get("best_macro_f1", state.get("macro_f1", -1.0))
            start_epoch = state["epoch"] + 1
            print(
                f"Resuming from {STATE_PATH}: epoch={state['epoch']}, "
                f"best_macro_f1={best_macro_f1*100:.2f}%"
            )
        except Exception as e:
            print(f"Warning: could not load {STATE_PATH}: {e}. Starting from scratch.")

    for epoch in range(start_epoch, EPOCHS + 1):
        base_model.eval()
        classifier.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - train", leave=False):
            images, labels = collate_fn(batch, DEVICE)

            optimizer.zero_grad()

            # Extract frozen video features, then train the classifier on top
            with torch.no_grad():
                feats = extract_video_features(base_model, images)  # [B, D]

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

        _, macro_f1 = evaluate(base_model, classifier, test_loader, DEVICE)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "classifier": classifier.state_dict(),
                    "macro_f1": best_macro_f1,
                    "epoch": epoch,
                },
                BEST_PATH,
            )
            print(f"Saved new best video-token model (macro-F1={macro_f1*100:.2f}%)")

        # Always save latest epoch state for resuming
        torch.save(
            {
                "classifier": classifier.state_dict(),
                "macro_f1": macro_f1,
                "best_macro_f1": best_macro_f1,
                "epoch": epoch,
            },
            STATE_PATH,
        )
        print(f"Saved last-epoch video-token model to {STATE_PATH}")


if __name__ == "__main__":
    main()


