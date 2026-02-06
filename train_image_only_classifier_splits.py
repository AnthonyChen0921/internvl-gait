import os
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from internvl_skeleton_adapter import InternVLWithSkeleton
from gavd_skeleton_dataset import GavdSkeletonDataset, TOP7_LABELS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 32
BATCH_SIZE = 1
EPOCHS = 20
LR = 5e-4
STATE_PATH = "image_only_train_state.pt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_samples_from_csv(csv_path: str, label_to_idx: Dict[str, int]) -> List[Dict]:
    df = pd.read_csv(csv_path)
    required = {"seq_id", "label", "skeleton_path", "video_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    samples: List[Dict] = []
    for _, r in df.iterrows():
        label = str(r["label"])
        if label not in label_to_idx:
            raise ValueError(f"Unknown label {label!r} in {csv_path}")
        samples.append(
            {
                "seq_id": str(r["seq_id"]),
                "path": str(r["skeleton_path"]),
                "video_path": str(r["video_path"]),  # per-sample override supported in GavdSkeletonDataset
                "label_str": label,
                "label_idx": int(label_to_idx[label]),
            }
        )
    return samples


def compute_class_weights(train_samples: List[Dict], labels: List[str]) -> torch.Tensor:
    counts = [0] * len(labels)
    for s in train_samples:
        counts[int(s["label_idx"])] += 1
    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    counts_tensor = torch.clamp(counts_tensor, min=1.0)
    N = counts_tensor.sum()
    K = float(len(labels))
    weights = N / (K * counts_tensor)
    weights = weights / weights.mean()
    print("\nClass weights:", weights.tolist())
    return weights


def collate_fn(batch, tokenizer, device):
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
        "- parkinsons: shuffling, stooped posture, reduced arm swing, and festination typical of Parkinson's disease.\n"
        "- dcm: controlled clinical gait trial videos (your DCM dataset).\n\n"
        "Answer by internally deciding which class is most likely; you do not need to output the class name."
    )
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    B = images.size(0)
    input_ids = input_ids.expand(B, -1).contiguous()
    attention_mask = attention_mask.expand(B, -1).contiguous()
    return images, labels, input_ids, attention_mask


def evaluate(model, classifier, data_loader, tokenizer, device, labels_list: List[str]):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0

    num_classes = len(labels_list)
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
            hidden = outputs.hidden_states[-1]
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
        per_class_f1.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    macro_f1 = sum(per_class_f1) / num_classes

    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"Test macro-F1: {macro_f1 * 100:.2f}%")
    print("Per-class F1:")
    for idx, f1 in enumerate(per_class_f1):
        print(f"  {labels_list[idx]}: {f1 * 100:.2f}%")
    return acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=str,
        required=True,
        help="Directory containing combined_train.csv and combined_test.csv (e.g. splits_gavd_plus_dcm_singlelabel).",
    )
    args = parser.parse_args()

    splits_dir = args.splits_dir if os.path.isabs(args.splits_dir) else os.path.join(BASE_DIR, args.splits_dir)
    train_csv = os.path.join(splits_dir, "combined_train.csv")
    test_csv = os.path.join(splits_dir, "combined_test.csv")

    labels_list = TOP7_LABELS + ["dcm"]  # single-label DCM mode
    label_to_idx = {lbl: i for i, lbl in enumerate(labels_list)}

    train_samples = load_samples_from_csv(train_csv, label_to_idx)
    test_samples = load_samples_from_csv(test_csv, label_to_idx)

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Test samples:  {len(test_samples)}")

    train_ds = GavdSkeletonDataset(train_samples, window_size=WINDOW_SIZE, train=True, with_images=True, video_dir=BASE_DIR)
    test_ds = GavdSkeletonDataset(test_samples, window_size=WINDOW_SIZE, train=False, with_images=True, video_dir=BASE_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    tokenizer, base_model, _ = load_model(device=DEVICE)
    img_model = InternVLWithSkeleton(base_model).to(DEVICE)
    img_model.eval()

    hidden_size = img_model.hidden_size
    classifier = nn.Linear(hidden_size, len(labels_list), dtype=torch.float32).to(DEVICE)
    class_weights = compute_class_weights(train_samples, labels_list).to(DEVICE)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR, weight_decay=1e-4)
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

        _, macro_f1 = evaluate(img_model, classifier, test_loader, tokenizer, DEVICE, labels_list)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "image_model": img_model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "macro_f1": best_macro_f1,
                    "epoch": epoch,
                },
                "best_image_only_classifier_plus_dcm.pt",
            )
            print(f"Saved new best model (macro-F1={macro_f1*100:.2f}%)")

        torch.save(
            {
                "image_model": img_model.state_dict(),
                "classifier": classifier.state_dict(),
                "macro_f1": macro_f1,
                "epoch": epoch,
            },
            STATE_PATH,
        )


if __name__ == "__main__":
    main()


