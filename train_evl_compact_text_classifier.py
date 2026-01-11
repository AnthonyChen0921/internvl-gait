import os
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from internvl_temporal_adapter import EVLTemporalDecoder
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

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GAVD-sequences")
CKPT_PATH = "best_evl_compact_text_classifier.pt"
STATE_PATH = "evl_compact_text_train_state.pt"


def build_dataloaders():
    samples = collect_labeled_sequences()
    train_samples, test_samples = video_level_train_test_split(samples, train_ratio=0.8)

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


def collate_fn(batch, tokenizer, device):
    """
    Batch from GavdSkeletonDataset with_images=True:
      {
        "skeleton": FloatTensor [B, W, 46],  # unused here
        "images":   FloatTensor [B, W, 3, H, W],
        "label":    LongTensor [B]
      }
    We keep images + labels and attach a fixed clinician prompt.
    """
    images = batch["images"].to(device)
    labels = batch["label"].to(device)

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

    B = images.size(0)
    input_ids = input_ids.expand(B, -1).contiguous()
    attention_mask = attention_mask.expand(B, -1).contiguous()

    return images, labels, input_ids, attention_mask


def evaluate(decoder, language_model, classifier, data_loader, tokenizer, device):
    decoder.eval()
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

            compact_tokens = decoder(pixel_values=images)  # [B, M, D]

            # Build LLM inputs: [compact video tokens] + [text tokens]
            text_embeds = language_model.get_input_embeddings()(input_ids)  # [B, L, D]
            compact_tokens = compact_tokens.to(dtype=text_embeds.dtype)

            B, M, _ = compact_tokens.shape
            video_mask = torch.ones(B, M, dtype=attention_mask.dtype, device=device)

            inputs_embeds = torch.cat([compact_tokens, text_embeds], dim=1)  # [B, M+L, D]
            fused_mask = torch.cat([video_mask, attention_mask], dim=1)  # [B, M+L]

            outputs = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=fused_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]  # [B, M+L, D]
            video_hidden = hidden[:, :M, :]  # compact video tokens after LLM
            feats = video_hidden.mean(dim=1).float()  # [B, D]

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
    language_model = getattr(base_model, "language_model", base_model)
    language_model.requires_grad_(False)

    decoder = EVLTemporalDecoder(
        base_model,
        max_frames=WINDOW_SIZE,
        num_queries=8,
        num_layers=3,
        num_heads=4,
    ).to(DEVICE)

    hidden_size = decoder.hidden_size
    num_classes = len(TOP7_LABELS)
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    train_loader, test_loader, train_samples = build_dataloaders()
    class_weights = compute_class_weights(train_samples).to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(decoder.parameters()) + list(classifier.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_macro_f1 = -1.0
    start_epoch = 1
    resumed_from_state = False

    if os.path.exists(STATE_PATH):
        try:
            state = torch.load(STATE_PATH, map_location=DEVICE)
            decoder.load_state_dict(state["decoder"])
            classifier.load_state_dict(state["classifier"])
            optimizer.load_state_dict(state["optimizer"])
            best_macro_f1 = state.get("best_macro_f1", -1.0)
            start_epoch = state.get("epoch", 0) + 1
            resumed_from_state = True
            print(
                f"Resuming from {STATE_PATH}: "
                f"epoch={state.get('epoch')}, best_macro_f1={best_macro_f1*100:.2f}%"
            )
        except Exception as e:
            print(f"Warning: failed to load {STATE_PATH} ({e}); deleting and falling back.")
            try:
                os.remove(STATE_PATH)
            except OSError:
                pass

    if (not resumed_from_state) and os.path.exists(CKPT_PATH):
        try:
            ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
            decoder.load_state_dict(ckpt["decoder"])
            classifier.load_state_dict(ckpt["classifier"])
            best_macro_f1 = ckpt.get("macro_f1", -1.0)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(
                f"Warm-starting from best checkpoint {CKPT_PATH}: "
                f"epoch={ckpt.get('epoch')}, best_macro_f1={best_macro_f1*100:.2f}%"
            )
        except Exception as e:
            print(f"Warning: failed to load best checkpoint {CKPT_PATH} ({e}); starting from scratch.")

    for epoch in range(start_epoch, EPOCHS + 1):
        decoder.train()
        classifier.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - train", leave=False):
            images, labels, input_ids, attention_mask = collate_fn(batch, tokenizer, DEVICE)

            optimizer.zero_grad()

            compact_tokens = decoder(pixel_values=images)  # [B, M, D]

            text_embeds = language_model.get_input_embeddings()(input_ids)  # [B, L, D]
            compact_tokens = compact_tokens.to(dtype=text_embeds.dtype)

            B, M, _ = compact_tokens.shape
            video_mask = torch.ones(B, M, dtype=attention_mask.dtype, device=DEVICE)

            inputs_embeds = torch.cat([compact_tokens, text_embeds], dim=1)  # [B, M+L, D]
            fused_mask = torch.cat([video_mask, attention_mask], dim=1)

            outputs = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=fused_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]  # [B, M+L, D]
            video_hidden = hidden[:, :M, :]
            feats = video_hidden.mean(dim=1).float()  # [B, D]

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

        _, macro_f1 = evaluate(decoder, language_model, classifier, test_loader, tokenizer, DEVICE)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "decoder": decoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "macro_f1": best_macro_f1,
                    "epoch": epoch,
                },
                CKPT_PATH,
            )
            print(f"Saved new best EVL-compact+text model (macro-F1={macro_f1*100:.2f}%)")

        torch.save(
            {
                "decoder": decoder.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_macro_f1": best_macro_f1,
            },
            STATE_PATH,
        )


if __name__ == "__main__":
    main()










