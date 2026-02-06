import os
import argparse
from typing import Dict, List, Tuple

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
EPOCHS = 40
LR = 5e-4

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GAVD-sequences")
CKPT_PATH = "best_evl_compact_text_classifier.pt"
STATE_PATH = "evl_compact_text_train_state.pt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARGS = None
LABELS: List[str] = []


def _print_label_distribution(samples: List[Dict], labels: List[str], title: str) -> None:
    counts = [0] * len(labels)
    for s in samples:
        idx = int(s["label_idx"])
        if 0 <= idx < len(counts):
            counts[idx] += 1
    parts = [f"{labels[i]}: {counts[i]}" for i in range(len(labels))]
    print(f"\n{title}: " + " ".join(parts))


def _load_samples_from_split_csv(csv_path: str, label_to_idx: Dict[str, int]) -> List[Dict]:
    """
    Load samples from a split CSV produced by `make_gavd_plus_dcm_splits.py`.
    Expected columns: seq_id,label,skeleton_path,video_path
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"seq_id", "label", "skeleton_path", "video_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV is missing columns {sorted(missing)}: {csv_path}")

    def _resolve(p: str) -> str:
        p = str(p)
        if not os.path.isabs(p):
            p = os.path.join(BASE_DIR, p)
        return os.path.normpath(p)

    samples: List[Dict] = []
    for _, r in df.iterrows():
        seq_id = str(r["seq_id"])
        label_str = str(r["label"])
        if label_str not in label_to_idx:
            raise ValueError(f"Unknown label {label_str!r} in {csv_path}. Known: {sorted(label_to_idx.keys())}")
        samples.append(
            {
                "seq_id": seq_id,
                "path": _resolve(r["skeleton_path"]),
                "video_path": _resolve(r["video_path"]),
                "label_str": label_str,
                "label_idx": int(label_to_idx[label_str]),
            }
        )
    return samples


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

    num_classes = len(LABELS)
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
        print(f"  {LABELS[idx]}: {f1 * 100:.2f}%")

    return acc, macro_f1


def main():
    global ARGS, LABELS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="",
        help="If set, loads combined_train.csv/combined_test.csv from this directory to train on GAVD+DCM.",
    )
    ARGS = parser.parse_args()
    if ARGS.splits_dir:
        ARGS.splits_dir = ARGS.splits_dir if os.path.isabs(ARGS.splits_dir) else os.path.join(BASE_DIR, ARGS.splits_dir)
        if not os.path.isdir(ARGS.splits_dir):
            raise FileNotFoundError(f"--splits-dir not found: {ARGS.splits_dir}")

    tokenizer, base_model, _ = load_model(device=DEVICE)
    language_model = getattr(base_model, "language_model", base_model)
    language_model.requires_grad_(False)

    decoder = EVLTemporalDecoder(
        base_model,
        max_frames=WINDOW_SIZE,
        num_queries=WINDOW_SIZE,  # align one query per frame
        num_layers=3,
        num_heads=4,
    ).to(DEVICE)

    hidden_size = decoder.hidden_size
    train_loader, test_loader, train_samples, LABELS = build_dataloaders()
    num_classes = len(LABELS)
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    class_weights = compute_class_weights(train_samples, labels=LABELS).to(DEVICE)

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














