import os
import csv
import glob
import argparse
from typing import Dict, List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from gavd_skeleton_dataset import (
    collect_labeled_sequences,
    video_level_train_test_split,
    TOP7_LABELS,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "GAVD-sequences")
RESULTS_PATH = os.path.join(BASE_DIR, "zero_shot_video_results.csv")
FRAMES_PER_SEQ = 16  # number of frames sampled per sequence


def load_video_window(seq_id: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Load a fixed number of frames from the video for the given sequence."""
    video_path = os.path.join(VIDEO_DIR, f"{seq_id}.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = FRAMES_PER_SEQ
    if frame_count <= 0:
        raise RuntimeError(f"No frames found in {video_path}")

    if frame_count <= W:
        indices = list(range(frame_count)) + [frame_count - 1] * (W - frame_count)
    else:
        step = frame_count / W
        indices = [int(i * step) for i in range(W)]

    transform = T.Compose(
        [
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frames = []
    last_valid = None
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            if last_valid is None:
                frame = np.zeros((448, 448, 3), dtype=np.uint8)
            else:
                frame = last_valid
        last_valid = frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        tensor = transform(pil)  # [3, 448, 448]
        frames.append(tensor)

    cap.release()

    pixel_values = torch.stack(frames, dim=0).to(device=device, dtype=dtype)  # [W, 3, 448, 448]
    return pixel_values


def load_existing_results() -> Dict[str, str]:
    """Return a mapping seq_id -> predicted_label from previous runs."""
    if not os.path.isfile(RESULTS_PATH):
        return {}
    preds: Dict[str, str] = {}
    with open(RESULTS_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds[row["seq_id"]] = row["pred_label"]
    return preds


def append_result(row: Dict[str, str]):
    """Append a single result row to the CSV (creating header if needed)."""
    file_exists = os.path.isfile(RESULTS_PATH)
    fieldnames = ["seq_id", "video_id", "true_label", "pred_label"]
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot InternVL video-only gait classification on GAVD sequences")
    parser.add_argument(
        "--split",
        choices=["test", "all", "train"],
        default="test",
        help="Which split of sequences to evaluate on: "
             "'test' (default, video-level test set), "
             "'train' (video-level train set), or 'all' (all labeled single-person sequences).",
    )
    args = parser.parse_args()

    tokenizer, model, device = load_model()

    print("Zero-shot video experiment: image sequence + text prompt (no skeleton).")

    # Determine model dtype
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32

    # Build labeled samples and video-level split
    samples = collect_labeled_sequences()
    train_samples, test_samples = video_level_train_test_split(samples, train_ratio=0.8)

    if args.split == "all":
        eval_samples = samples
        split_name = "all sequences"
    elif args.split == "train":
        eval_samples = train_samples
        split_name = "train split"
    else:
        eval_samples = test_samples
        split_name = "test split"

    print(f"Total labeled sequences: {len(samples)}")
    print(f"Evaluating on {len(eval_samples)} sequences ({split_name}).")

    # Load already processed results to support resume
    existing = load_existing_results()
    eval_seq_ids = {s["seq_id"] for s in eval_samples}
    processed = {sid for sid in existing.keys() if sid in eval_seq_ids}
    print(f"Existing results for {len(processed)} sequences in this split will be skipped.")

    # Define expert gait prompt
    prompt = (
        "You are an expert gait clinician. Based on the sequence of gait images, "
        "classify the patient's gait pattern.\n\n"
        "Gait pattern definitions:\n"
        "- abnormal: any gait pattern that deviates from normal but does not fit the specific patterns below.\n"
        "- myopathic: waddling or Trendelenburg-type gait due to proximal muscle weakness.\n"
        "- exercise: exaggerated, energetic, or performance-like gait related to sport or exercise.\n"
        "- normal: typical, symmetric gait without obvious abnormalities.\n"
        "- style: exaggerated or stylistic walking pattern without clear neurological or orthopedic cause.\n"
        "- cerebral palsy: spastic, scissoring, toe-walking, or crouched gait typical of cerebral palsy.\n"
        "- parkinsons: shuffling, stooped posture, reduced arm swing, and festination typical of Parkinson's disease.\n\n"
        "Based solely on this image sequence, what is the most likely gait pattern?\n"
        "Just answer with ONE of the following class names exactly:\n"
        "abnormal, myopathic, exercise, normal, style, cerebral palsy, parkinsons."
    )

    # Iterate over test samples, skipping those already processed
    all_true: List[int] = []
    all_pred: List[int] = []

    from collections import Counter

    for meta in tqdm(eval_samples, desc="Zero-shot eval", unit="seq"):
        seq_id = meta["seq_id"]
        if seq_id in processed:
            continue

        true_label_idx = meta["label_idx"]
        true_label_str = meta["label_str"]

        try:
            pixel_values = load_video_window(seq_id, device, model_dtype)  # [W, 3, 448, 448]
        except Exception as e:
            print(f"Warning: skipping {seq_id} due to video error: {e}")
            continue

        # Run zero-shot prediction
        with torch.no_grad():
            try:
                response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config={"max_new_tokens": 64, "do_sample": False},
                )
            except Exception as e:
                print(f"Warning: model.chat failed for {seq_id}: {e}")
                continue

        # Extract the first matching label from response
        resp_lower = response.lower()
        pred_label_str = None
        for label in TOP7_LABELS:
            if label in resp_lower:
                pred_label_str = label
                break

        if pred_label_str is None:
            print(f"Warning: no label found in response for {seq_id}: {response!r}")
            continue

        pred_label_idx = TOP7_LABELS.index(pred_label_str)

        # Record result
        append_result(
            {
                "seq_id": seq_id,
                "video_id": meta["video_id"],
                "true_label": true_label_str,
                "pred_label": pred_label_str,
            }
        )

        all_true.append(true_label_idx)
        all_pred.append(pred_label_idx)

    # If we just resumed and didn't add new predictions, recompute from CSV for this split
    if existing and not all_true:
        with open(RESULTS_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["seq_id"] not in eval_seq_ids:
                    continue
                all_true.append(TOP7_LABELS.index(row["true_label"]))
                all_pred.append(TOP7_LABELS.index(row["pred_label"]))

    # Compute and print metrics
    if not all_true:
        print("No predictions available to evaluate.")
        return

    assert len(all_true) == len(all_pred)
    total = len(all_true)
    correct = sum(int(t == p) for t, p in zip(all_true, all_pred))
    acc = correct / total

    num_classes = len(TOP7_LABELS)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for t, p in zip(all_true, all_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class_f1 = []
    for c in range(num_classes):
        prec = tp[c] / max(1, tp[c] + fp[c])
        rec = tp[c] / max(1, tp[c] + fn[c])
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        per_class_f1.append(f1)

    macro_f1 = sum(per_class_f1) / num_classes

    print(f"\n=== Zero-shot video-only results ({split_name}) ===")
    print(f"Sequences evaluated: {total}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro-F1: {macro_f1 * 100:.2f}%")
    print("Per-class F1:")
    for idx, f1 in enumerate(per_class_f1):
        print(f"  {TOP7_LABELS[idx]}: {f1 * 100:.2f}%")


if __name__ == "__main__":
    main()



