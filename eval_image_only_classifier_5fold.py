import os
import random
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
    TOP7_LABELS,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 32
BATCH_SIZE = 1
NUM_FOLDS = 5
RANDOM_SEED = 42

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GAVD-sequences")
FOLD_CKPT_TEMPLATE = "best_image_only_classifier_fold{fold}.pt"
EVAL_RESULTS_PATH = "image_only_5fold_eval_results.log"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def video_level_kfold_split(samples: List[Dict], num_folds: int) -> List[Tuple[List[Dict], List[Dict]]]:
    by_label_videos: Dict[int, List[str]] = {}
    for s in samples:
        by_label_videos.setdefault(s["label_idx"], set()).add(s["video_id"])

    for label_idx, vids in by_label_videos.items():
        vid_list = list(vids)
        random.shuffle(vid_list)
        by_label_videos[label_idx] = vid_list

    folds: List[set] = [set() for _ in range(num_folds)]
    for label_idx, vid_list in by_label_videos.items():
        for i, vid in enumerate(vid_list):
            folds[i % num_folds].add(vid)

    splits: List[Tuple[List[Dict], List[Dict]]] = []
    for fold_idx in range(num_folds):
        test_videos = folds[fold_idx]
        train_videos = set().union(*[folds[i] for i in range(num_folds) if i != fold_idx])
        train_samples = [s for s in samples if s["video_id"] in train_videos]
        test_samples = [s for s in samples if s["video_id"] in test_videos]
        splits.append((train_samples, test_samples))
    return splits


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


def append_eval_result(fold_idx: int, acc: float, macro_f1: float) -> None:
    with open(EVAL_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(
            f"fold{fold_idx}: acc={acc * 100:.2f}%, macro_f1={macro_f1 * 100:.2f}%\n"
        )


def main():
    set_seed(RANDOM_SEED)
    if os.path.exists(EVAL_RESULTS_PATH):
        os.remove(EVAL_RESULTS_PATH)

    samples = collect_labeled_sequences()
    splits = video_level_kfold_split(samples, NUM_FOLDS)

    tokenizer, base_model, _ = load_model(device=DEVICE)
    img_model = InternVLWithSkeleton(base_model).to(DEVICE)
    img_model.eval()

    hidden_size = img_model.hidden_size
    num_classes = len(TOP7_LABELS)

    for fold_idx, (_, test_samples) in enumerate(splits):
        ckpt_path = FOLD_CKPT_TEMPLATE.format(fold=fold_idx)
        if not os.path.exists(ckpt_path):
            print(f"Skipping fold {fold_idx}: missing checkpoint {ckpt_path}")
            continue

        classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        img_model.load_state_dict(ckpt["image_model"])
        classifier.load_state_dict(ckpt["classifier"])
        print(
            f"Loaded fold {fold_idx} checkpoint from {ckpt_path} "
            f"(epoch {ckpt.get('epoch')}, macro-F1={ckpt.get('macro_f1', 0.0) * 100:.2f}%)"
        )

        test_ds = GavdSkeletonDataset(
            test_samples,
            window_size=WINDOW_SIZE,
            train=False,
            with_images=True,
            video_dir=VIDEO_DIR,
        )
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        acc, macro_f1 = evaluate(img_model, classifier, test_loader, tokenizer, DEVICE)
        append_eval_result(fold_idx, acc, macro_f1)


if __name__ == "__main__":
    main()

