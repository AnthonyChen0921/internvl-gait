import os
import re
import argparse
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image

from transformers import AutoProcessor, AutoModelForVision2Seq

from gavd_skeleton_dataset import GavdSkeletonDataset, TOP7_LABELS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 32
BATCH_SIZE = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARGS = None
IMAGE_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


def _resolve(p: str) -> str:
    p = str(p)
    if not os.path.isabs(p):
        p = os.path.join(BASE_DIR, p)
    return os.path.normpath(p)


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


def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Always uses GAVD+DCM split CSVs. Returns train, test, and combined loaders.
    """
    assert ARGS is not None
    if not ARGS.splits_dir:
        raise ValueError("--splits-dir is required for GAVD+DCM evaluation.")

    splits_dir = ARGS.splits_dir
    train_csv = os.path.join(splits_dir, "combined_train.csv")
    test_csv = os.path.join(splits_dir, "combined_test.csv")

    labels = TOP7_LABELS + ["dcm"]
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    train_samples = _load_samples_from_split_csv(train_csv, label_to_idx=label_to_idx)
    test_samples = _load_samples_from_split_csv(test_csv, label_to_idx=label_to_idx)
    all_samples = train_samples + test_samples

    print(f"\nTrain sequences: {len(train_samples)}")
    print(f"Test sequences: {len(test_samples)}")
    print(f"All sequences: {len(all_samples)}")

    train_ds = GavdSkeletonDataset(
        train_samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=BASE_DIR,
    )
    test_ds = GavdSkeletonDataset(
        test_samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=BASE_DIR,
    )
    all_ds = GavdSkeletonDataset(
        all_samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=BASE_DIR,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_loader = DataLoader(all_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader, all_loader, labels


def build_prompt(labels: List[str]) -> str:
    return (
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
        "- dcm: dilated cardiomyopathy or related DCM gait patterns.\n\n"
        "Answer with a single label from this list only: "
        f"{', '.join(labels)}.\n"
        "Label:"
    )


def _denormalize_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """
    Convert normalized [T, 3, H, W] tensor to list of PIL RGB images.
    Dataset uses ImageNet normalization with mean/std.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    imgs = images * std + mean
    imgs = imgs.clamp(0, 1)
    imgs = (imgs * 255).to(torch.uint8).cpu()

    pil_list: List[Image.Image] = []
    for i in range(imgs.size(0)):
        arr = imgs[i].permute(1, 2, 0).numpy()
        pil_list.append(Image.fromarray(arr, mode="RGB"))
    return pil_list


def _normalize_label(text: str, labels: List[str]) -> str:
    clean = text.lower().strip()
    if "label:" in clean:
        clean = clean.split("label:", 1)[1].strip()
    aliases = {
        "parkinson": "parkinsons",
        "parkinson's": "parkinsons",
        "cerebral palsy": "cerebral palsy",
        "cp": "cerebral palsy",
        "dilated cardiomyopathy": "dcm",
    }

    for lbl in labels:
        if re.search(rf"\b{re.escape(lbl)}\b", clean):
            return lbl
    for key, val in aliases.items():
        if key in clean and val in labels:
            return val
    return ""


def _prepare_inputs(processor, seq_id: str, images: torch.Tensor, prompt: str) -> Dict[str, torch.Tensor]:
    if ARGS.cache_vision and seq_id in IMAGE_CACHE:
        cached = IMAGE_CACHE[seq_id]
        return {k: v.clone() for k, v in cached.items()}

    pil_images = _denormalize_to_pil(images)
    if ARGS.max_frames and len(pil_images) > ARGS.max_frames:
        idxs = torch.linspace(0, len(pil_images) - 1, ARGS.max_frames).long().tolist()
        pil_images = [pil_images[i] for i in idxs]

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in pil_images]
            + [{"type": "text", "text": prompt}],
        }
    ]

    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
    else:
        text = prompt

    inputs = processor(
        text=[text],
        images=pil_images,
        return_tensors="pt",
        padding=True,
    )

    if ARGS.cache_vision:
        IMAGE_CACHE[seq_id] = {k: v.detach().cpu() for k, v in inputs.items()}
    return inputs


@torch.no_grad()
def predict_label(model, processor, batch: Dict, labels: List[str]) -> Tuple[str, str]:
    images = batch["images"].squeeze(0)
    seq_id = batch["seq_id"][0] if "seq_id" in batch else "unknown"
    prompt = build_prompt(labels)

    inputs = _prepare_inputs(processor, str(seq_id), images, prompt)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=ARGS.max_new_tokens,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    raw_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    pred = _normalize_label(raw_text, labels)
    return pred, raw_text


def evaluate(model, processor, data_loader, labels: List[str], title: str):
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    correct = 0
    total = 0

    num_classes = len(labels)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for batch in tqdm(data_loader, desc=f"Evaluating ({title})", leave=False):
        true_idx = int(batch["label"][0].item())
        true_label = labels[true_idx]

        pred_label, raw_text = predict_label(model, processor, batch, labels)
        if ARGS.print_output:
            seq_id = batch["seq_id"][0] if "seq_id" in batch else "unknown"
            print(f"\n[{title}] seq_id={seq_id}")
            print(f"raw_output: {raw_text}")
            print(f"pred_label: {pred_label if pred_label else 'N/A'} | true_label: {true_label}")

        if pred_label == "":
            pred_label = ARGS.fallback_label

        pred_idx = label_to_idx[pred_label]

        correct += int(pred_idx == true_idx)
        total += 1

        for c in range(num_classes):
            tp[c] += int(pred_idx == c and true_idx == c)
            fp[c] += int(pred_idx == c and true_idx != c)
            fn[c] += int(pred_idx != c and true_idx == c)

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

    print(f"\n{title} accuracy: {acc * 100:.2f}%")
    print(f"{title} macro-F1: {macro_f1 * 100:.2f}%")
    print(f"{title} per-class F1:")
    for idx, f1 in enumerate(per_class_f1):
        print(f"  {labels[idx]}: {f1 * 100:.2f}%")

    return acc, macro_f1


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Hugging Face model id for Qwen-VL.",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        required=True,
        help="Directory containing combined_train.csv and combined_test.csv.",
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Print raw model output and parsed label for each sample.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens for zero-shot label generation.",
    )
    parser.add_argument(
        "--fallback-label",
        type=str,
        default="abnormal",
        help="Label to use when parsing fails.",
    )
    parser.add_argument(
        "--cache-vision",
        action="store_true",
        help="Cache preprocessed vision inputs by seq_id to speed up repeated evaluation.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=WINDOW_SIZE,
        help="Max frames to feed into Qwen-VL (uniformly sampled if needed).",
    )
    ARGS = parser.parse_args()
    ARGS.splits_dir = _resolve(ARGS.splits_dir)
    if not os.path.isdir(ARGS.splits_dir):
        raise FileNotFoundError(f"--splits-dir not found: {ARGS.splits_dir}")

    processor = AutoProcessor.from_pretrained(ARGS.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        ARGS.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()

    train_loader, test_loader, all_loader, labels = build_dataloaders()
    evaluate(model, processor, test_loader, labels, title="Test")
    evaluate(model, processor, all_loader, labels, title="All (train+test)")


if __name__ == "__main__":
    main()
