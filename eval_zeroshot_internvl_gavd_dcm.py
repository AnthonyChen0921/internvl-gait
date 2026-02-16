import os
import re
import argparse
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from gavd_skeleton_dataset import GavdSkeletonDataset, TOP7_LABELS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 32
BATCH_SIZE = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "GAVD-sequences")

ARGS = None


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
        "- dcm: dilated cardiomyopathy or related DCM gait patterns.\n\n"
        "Answer with a single label from this list only: "
        f"{', '.join(labels)}.\n"
        "Label:"
    )
    return prompt


def _inject_visual_tokens(model, tokenizer, pixel_values: torch.Tensor, prompt: str):
    """
    Build input embeddings by inserting visual tokens at IMG_CONTEXT positions.
    """
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    question = "<image>\n" + prompt

    num_frames = pixel_values.shape[0]
    num_image_token = getattr(model, "num_image_token", 1)
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (num_image_token * num_frames) + IMG_END_TOKEN
    query = question.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(DEVICE)
    attention_mask = model_inputs["attention_mask"].to(DEVICE)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    vit_embeds = model.extract_feature(pixel_values)  # [num_frames, T_v, D]
    language_model = getattr(model, "language_model", model)
    input_embeds = language_model.get_input_embeddings()(input_ids)

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)
    flat_ids = input_ids.reshape(B * N)
    selected = flat_ids == img_context_token_id

    vit_flat = vit_embeds.reshape(-1, C).to(input_embeds.device)
    if vit_flat.size(0) >= selected.sum():
        input_embeds[selected] = vit_flat[: selected.sum()]
    else:
        input_embeds[selected.nonzero(as_tuple=True)[0][: vit_flat.size(0)]] = vit_flat

    input_embeds = input_embeds.reshape(B, N, C)
    return input_embeds, attention_mask


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


@torch.no_grad()
def _label_token_ids(tokenizer, label: str) -> torch.Tensor:
    label_text = " " + label
    enc = tokenizer(label_text, add_special_tokens=False, return_tensors="pt")
    label_ids = enc["input_ids"]
    if label_ids.numel() == 0:
        label_ids = tokenizer(label, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return label_ids


@torch.no_grad()
def _score_labels(model, tokenizer, prompt_embeds, prompt_mask, labels: List[str]) -> Dict[str, float]:
    language_model = getattr(model, "language_model", model)
    scores: Dict[str, float] = {}

    prompt_len = prompt_embeds.size(1)
    for label in labels:
        label_ids = _label_token_ids(tokenizer, label).to(DEVICE)
        label_len = label_ids.size(1)
        label_embeds = language_model.get_input_embeddings()(label_ids)

        inputs_embeds = torch.cat([prompt_embeds, label_embeds], dim=1)
        label_mask = torch.ones(1, label_len, dtype=prompt_mask.dtype, device=prompt_mask.device)
        attention_mask = torch.cat([prompt_mask, label_mask], dim=1)

        outputs = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits  # [1, seq_len, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

        total = 0.0
        for i in range(label_len):
            pos = prompt_len - 1 + i
            token_id = int(label_ids[0, i].item())
            total += float(log_probs[0, pos, token_id].item())
        scores[label] = total / max(1, label_len)
    return scores


@torch.no_grad()
def predict_label(model, tokenizer, images: torch.Tensor, labels: List[str]) -> Tuple[str, str]:
    pixel_values = images.squeeze(0)
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    pixel_values = pixel_values.to(device=DEVICE, dtype=model_dtype)

    prompt = build_prompt(labels)
    inputs_embeds, attention_mask = _inject_visual_tokens(model, tokenizer, pixel_values, prompt)
    language_model = getattr(model, "language_model", model)

    scores = _score_labels(model, tokenizer, inputs_embeds, attention_mask, labels)
    pred = max(scores, key=scores.get) if scores else ""

    raw_text = ""
    if ARGS.print_output:
        output_ids = language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=ARGS.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return pred, raw_text


def evaluate(model, tokenizer, data_loader, labels: List[str], title: str):
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    correct = 0
    total = 0

    num_classes = len(labels)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for batch in tqdm(data_loader, desc=f"Evaluating ({title})", leave=False):
        images = batch["images"]
        true_label = batch["label_str"][0] if "label_str" in batch else None
        if true_label is None:
            true_idx = int(batch["label"][0].item())
            true_label = labels[true_idx]

        pred_label, raw_text = predict_label(model, tokenizer, images, labels)
        if ARGS.print_output:
            seq_id = batch["seq_id"][0] if "seq_id" in batch else "unknown"
            print(f"\n[{title}] seq_id={seq_id}")
            print(f"raw_output: {raw_text}")
            print(f"pred_label: {pred_label if pred_label else 'N/A'} | true_label: {true_label}")
        if pred_label == "":
            pred_label = ARGS.fallback_label

        pred_idx = label_to_idx[pred_label]
        true_idx = label_to_idx[true_label]

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
    ARGS = parser.parse_args()
    ARGS.splits_dir = _resolve(ARGS.splits_dir)
    if not os.path.isdir(ARGS.splits_dir):
        raise FileNotFoundError(f"--splits-dir not found: {ARGS.splits_dir}")

    tokenizer, model, _ = load_model(device=DEVICE)
    model.to(DEVICE)
    model.eval()

    train_loader, test_loader, all_loader, labels = build_dataloaders()
    evaluate(model, tokenizer, test_loader, labels, title="Test")
    evaluate(model, tokenizer, all_loader, labels, title="All (train+test)")


if __name__ == "__main__":
    main()
