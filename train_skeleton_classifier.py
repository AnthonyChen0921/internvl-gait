import os
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
WINDOW_SIZE = 64
BATCH_SIZE = 2
EPOCHS = 20
LR = 5e-4

CKPT_STATE_PATH = "skeleton_train_state.pt"


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

    train_ds = GavdSkeletonDataset(train_samples, window_size=WINDOW_SIZE, train=True)
    test_ds = GavdSkeletonDataset(test_samples, window_size=WINDOW_SIZE, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_samples


def compute_class_weights(train_samples: List[Dict]) -> torch.Tensor:
    counts = [0] * len(TOP7_LABELS)
    for s in train_samples:
        counts[s["label_idx"]] += 1

    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    # Avoid division by zero
    counts_tensor = torch.clamp(counts_tensor, min=1.0)
    N = counts_tensor.sum()
    K = float(len(TOP7_LABELS))
    weights = N / (K * counts_tensor)
    weights = weights / weights.mean()
    print("\nClass weights:", weights.tolist())
    return weights


def collate_fn(batch, tokenizer, device):
    """
    Prepare tensors for a batch.

    `batch` is the default-collated output from DataLoader, i.e. a dict:
      {
        'skeleton': FloatTensor [B, W, 46],
        'label': LongTensor [B]
      }
    """
    skeletons = batch["skeleton"].to(device)  # [B, W, 46]
    labels = batch["label"].to(device)        # [B]

    # Fixed text prompt for all samples.
    # We define each class explicitly to give the LM rich semantic context.
    prompt = (
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
        "Answer by internally deciding which class is most likely; you do not need to output the class name."
    )
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Expand to batch size
    B = skeletons.size(0)
    input_ids = input_ids.expand(B, -1).contiguous()
    attention_mask = attention_mask.expand(B, -1).contiguous()

    return skeletons, labels, input_ids, attention_mask


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
            skeletons, labels, input_ids, attention_mask = collate_fn(batch, tokenizer, device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                skeleton_feats=skeletons,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]  # [B, W+L, D]
            skel_hidden = hidden[:, :WINDOW_SIZE, :]
            feats = skel_hidden.mean(dim=1).float()  # [B, D] in float32

            logits = classifier(feats)  # classifier is float32
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for c in range(num_classes):
                tp[c] += ((preds == c) & (labels == c)).sum().item()
                fp[c] += ((preds == c) & (labels != c)).sum().item()
                fn[c] += ((preds != c) & (labels == c)).sum().item()

    acc = correct / max(1, total)

    # Per-class precision, recall, F1
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

    # Wrap with skeleton adapter, keep InternVL frozen
    skel_model = InternVLWithSkeleton(base_model).to(DEVICE)
    skel_model.eval()  # no dropout in frozen backbone

    hidden_size = skel_model.hidden_size
    num_classes = len(TOP7_LABELS)

    # Use float32 for the classifier for numerical stability
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    train_loader, test_loader, train_samples = build_dataloaders()
    class_weights = compute_class_weights(train_samples).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(skel_model.parameters()) + list(classifier.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    # Freeze base InternVL explicitly, in case
    skel_model.base_model.requires_grad_(False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # === Resume support ===
    start_epoch = 1
    best_macro_f1 = -1.0
    if os.path.isfile(CKPT_STATE_PATH):
        print(f"Found existing training state at {CKPT_STATE_PATH}, attempting to resume...")
        try:
            state = torch.load(CKPT_STATE_PATH, map_location=DEVICE)
            skel_model.load_state_dict(state["skel_model"])
            classifier.load_state_dict(state["classifier"])
            optimizer.load_state_dict(state["optimizer"])
            best_macro_f1 = state.get("best_macro_f1", -1.0)
            start_epoch = state.get("epoch", 0) + 1
            print(f"Resuming from epoch {start_epoch} with best_macro_f1={best_macro_f1*100:.2f}%")
        except Exception as e:
            # Corrupted or incompatible state file; delete it and start fresh.
            print(f"Warning: failed to load {CKPT_STATE_PATH} ({e}); deleting and starting from scratch.")
            try:
                os.remove(CKPT_STATE_PATH)
            except OSError:
                pass

    for epoch in range(start_epoch, EPOCHS + 1):
        skel_model.train()
        classifier.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - train", leave=False):
            skeletons, labels, input_ids, attention_mask = collate_fn(batch, tokenizer, DEVICE)

            optimizer.zero_grad()

            outputs = skel_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                skeleton_feats=skeletons,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]  # [B, W+L, D]
            skel_hidden = hidden[:, :WINDOW_SIZE, :]
            feats = skel_hidden.mean(dim=1).float()  # [B, D] in float32

            logits = classifier(feats)
            loss = criterion(logits, labels)
            loss.backward()

            # Only adapter + classifier should have gradients
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        print(f"\nEpoch {epoch}/{EPOCHS} - train loss: {train_loss:.4f}, acc: {train_acc*100:.2f}%")

        # Evaluate on test set each epoch
        _, macro_f1 = evaluate(skel_model, classifier, test_loader, tokenizer, DEVICE)

        # Save best model by macro-F1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_path = "best_skeleton_classifier.pt"
            torch.save(
                {
                    "skel_model": skel_model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "macro_f1": best_macro_f1,
                    "epoch": epoch,
                },
                save_path,
            )
            print(f"Saved new best model (macro-F1={macro_f1*100:.2f}%) to {save_path}")

        # Save training state for resume
        torch.save(
            {
                "skel_model": skel_model.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_macro_f1": best_macro_f1,
                "epoch": epoch,
            },
            CKPT_STATE_PATH,
        )


if __name__ == "__main__":
    main()


