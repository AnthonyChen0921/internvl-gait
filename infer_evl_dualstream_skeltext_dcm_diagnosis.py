import os
import json
import argparse
from typing import Dict, List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from internvl_skeleton_adapter import InternVLWithSkeleton
from internvl_temporal_adapter import EVLTemporalDecoder
from gavd_skeleton_dataset import GavdSkeletonDataset, TOP7_LABELS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 32
BATCH_SIZE = 1
MAX_TEXT_TOKENS = 1024
MAX_SKELETON_TEXT_FRAMES = 16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(p: str) -> str:
    p = str(p)
    if not os.path.isabs(p):
        p = os.path.join(BASE_DIR, p)
    return os.path.normpath(p)


def _load_samples_from_csv(csv_path: str, label_to_idx: Dict[str, int]) -> List[Dict]:
    df = pd.read_csv(csv_path)
    required = {"seq_id", "label", "skeleton_path", "video_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV is missing columns {sorted(missing)}: {csv_path}")

    has_text = "text_path" in df.columns
    out: List[Dict] = []
    for _, r in df.iterrows():
        label_str = str(r["label"])
        if label_str not in label_to_idx:
            raise ValueError(f"Unknown label {label_str!r} in {csv_path}")
        item = {
            "seq_id": str(r["seq_id"]),
            "path": _resolve(r["skeleton_path"]),
            "video_path": _resolve(r["video_path"]),
            "label_str": label_str,
            "label_idx": int(label_to_idx[label_str]),
        }
        if has_text and isinstance(r["text_path"], str) and r["text_path"].strip():
            item["text_path"] = _resolve(r["text_path"])
        out.append(item)
    return out


def _load_gait_json_as_text(gait_json_path: str) -> str:
    if not os.path.exists(gait_json_path):
        return "(gait parameter json not found)"
    try:
        with open(gait_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return "(failed to parse gait parameter json)"

    num_frames = obj.get("num_frames", "unknown")
    fps = obj.get("fps", "unknown")
    events = obj.get("events", {})
    counts = events.get("counts", {})
    metrics = obj.get("metrics", {})

    payload = {
        "num_frames": num_frames,
        "fps": fps,
        "event_counts": counts if isinstance(counts, dict) else {},
        "metrics": metrics if isinstance(metrics, dict) else {},
    }
    # Keep all available values with original precision for diagnosis prompting.
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _load_skeleton_text(text_path: str, start: int, window_size: int) -> str:
    if not isinstance(text_path, str) or text_path.strip() == "" or not os.path.exists(text_path):
        return ""

    records = []
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(obj)
    except OSError:
        return ""

    if not records:
        return ""

    max_idx = len(records) - 1
    num_available = min(window_size, len(records) - start if start <= max_idx else 0)
    if num_available <= 0:
        num_available = min(len(records), window_size)

    num_lines = min(num_available, MAX_SKELETON_TEXT_FRAMES)
    if num_lines <= 0:
        return ""

    step = max(1, window_size // num_lines)
    lines: List[str] = []
    used = 0
    frame_idx = start
    while used < num_lines and frame_idx <= start + window_size - 1:
        idx = min(frame_idx, max_idx)
        rec = records[idx]
        skel_str = rec.get("skel", "")
        frame_no = rec.get("frame", idx)
        if skel_str:
            lines.append(f"Frame {frame_no}: {skel_str}")
            used += 1
        frame_idx += step

    return "\n".join(lines)


def _build_classification_prompt(skeleton_text: str) -> str:
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
        "- dcm: degenerative cervical myelopathy-related gait impairment.\n\n"
        "Below are per-frame skeleton parameters extracted from the same gait sequence.\n"
        "Use both the visual gait information and these skeleton parameters to internally decide which class is most likely. "
        "You do not need to output the class name.\n\n"
        "Skeleton parameters:\n"
        f"{skeleton_text if skeleton_text else '(No skeleton parameters available for this sequence.)'}"
    )


def _extract_image_text_feats(img_model, images, input_ids, attention_mask):
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
        feats_img = img_hidden.mean(dim=1).float()
    return feats_img


def _extract_qm_feats(decoder, images):
    qm_tokens = decoder(pixel_values=images)
    return qm_tokens.mean(dim=1).float()


def _make_diagnosis_prompt(
    seq_id: str,
    pred_label: str,
    pred_confidence: float,
    top3_text: str,
    gait_text: str,
    user_instruction: str,
) -> str:
    pred_label_hint = (
        "Important: The classifier output above is already the final class decision for this sample. "
        "Do not replace it with another class."
    )
    if pred_label == "dcm":
        pred_label_hint += (
            " For this case, treat the diagnosis class as dcm and provide a dcm-consistent gait interpretation. "
            "Do not output 'normal' as the final diagnosis class."
        )
    return (
        f"{user_instruction}\n\n"
        f"Sequence ID: {seq_id}\n"
        "Classifier output (from trained model):\n"
        f"- predicted_label: {pred_label}\n"
        f"- predicted_confidence: {pred_confidence:.4f}\n"
        f"- top3: {top3_text}\n"
        f"{pred_label_hint}\n"
        "Skeleton gait parameters JSON (use numeric values exactly as provided):\n"
        f"{gait_text}\n\n"
        "Output format (strict):\n"
        f"1) Final diagnosis class: {pred_label}\n"
        "2) Rationale: 3-5 concise clinical sentences using the gait parameters.\n"
        "3) Confidence statement aligned with predicted_confidence."
    )


def _extract_final_diagnosis_class(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lower = text.lower()
    marker = "final diagnosis class:"
    if marker not in lower:
        return ""
    after = lower.split(marker, 1)[1].strip()
    if not after:
        return ""
    token = after.splitlines()[0].strip().split()[0].strip(" .,:;()[]{}")
    return token


@torch.no_grad()
def _generate_diagnosis(base_model, tokenizer, images_batched: torch.Tensor, prompt: str, max_new_tokens: int) -> str:
    # Use one representative frame from the input video window.
    center_idx = images_batched.shape[1] // 2
    frame = images_batched[:, center_idx, :, :, :]

    model_dtype = getattr(base_model, "dtype", None)
    if model_dtype is None:
        try:
            model_dtype = next(base_model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
    frame = frame.to(device=DEVICE, dtype=model_dtype)

    try:
        out = base_model.chat(
            tokenizer=tokenizer,
            pixel_values=frame,
            question=prompt,
            generation_config={"max_new_tokens": max_new_tokens, "do_sample": False},
        )
        if isinstance(out, str):
            return out.strip()
        return str(out).strip()
    except Exception:
        # Fallback to text-only if the chat API changes.
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        gen_model = getattr(base_model, "language_model", base_model)
        output = gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-csv",
        type=str,
        default="splits/gavd_plus_dcm_singlelabel_legacygavd/dcm_train.csv",
        help="CSV containing DCM split samples (e.g., dcm_train.csv or dcm_test.csv).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_evl_dualstream_skeltext_classifier.pt",
        help="Path to trained checkpoint that contains decoder + classifier states.",
    )
    parser.add_argument(
        "--gait-json-dir",
        type=str,
        default="gait_parameter_results",
        help="Directory containing HSMR-<seq_id>.gait.json files.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="You are a gait diagnosis assistant. Use the video evidence and provided gait parameters to generate diagnosis.",
        help="Instruction prompt for diagnosis generation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Max new tokens for diagnosis generation.")
    parser.add_argument("--max-samples", type=int, default=0, help="If > 0, evaluate only first N samples.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="reports/dcm_dualstream_diagnosis_on_dcm_train.csv",
        help="Where to save per-sample predictions and diagnoses.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/dcm_dualstream_diagnosis_on_dcm_train_summary.json",
        help="Where to save summary metrics.",
    )
    args = parser.parse_args()

    split_csv = _resolve(args.split_csv)
    ckpt_path = _resolve(args.checkpoint)
    gait_json_dir = _resolve(args.gait_json_dir)
    output_csv = _resolve(args.output_csv)
    output_json = _resolve(args.output_json)

    if not os.path.isfile(split_csv):
        raise FileNotFoundError(f"--split-csv not found: {split_csv}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"--checkpoint not found: {ckpt_path}")
    if not os.path.isdir(gait_json_dir):
        alt = _resolve("gait_parameter_result")
        if os.path.isdir(alt):
            gait_json_dir = alt
        else:
            raise FileNotFoundError(
                f"--gait-json-dir not found: {gait_json_dir} (also checked {alt})"
            )

    labels = TOP7_LABELS + ["dcm"]
    label_to_idx = {k: i for i, k in enumerate(labels)}
    samples = _load_samples_from_csv(split_csv, label_to_idx=label_to_idx)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    print(f"Samples to run: {len(samples)} from {split_csv}")
    dset = GavdSkeletonDataset(
        samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=BASE_DIR,
    )
    loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    tokenizer, base_model, _ = load_model(device=DEVICE)
    base_model.requires_grad_(False)
    img_model = InternVLWithSkeleton(base_model).to(DEVICE)
    img_model.eval()
    decoder = EVLTemporalDecoder(
        base_model,
        max_frames=WINDOW_SIZE,
        num_queries=WINDOW_SIZE,
        num_layers=3,
        num_heads=4,
    ).to(DEVICE)
    decoder.eval()

    hidden_size = img_model.hidden_size
    classifier = nn.Linear(hidden_size * 2, len(labels), dtype=torch.float32).to(DEVICE)
    classifier.eval()

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if "decoder" not in ckpt or "classifier" not in ckpt:
        raise ValueError(f"Checkpoint does not contain decoder/classifier keys: {ckpt_path}")
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    classifier.load_state_dict(ckpt["classifier"], strict=True)

    records: List[Dict] = []
    dcm_idx = label_to_idx["dcm"]
    correct = 0
    total = 0
    correct_dcm_binary = 0

    for batch in tqdm(loader, desc="Infer DCM", leave=False):
        seq_id = batch["seq_id"][0]
        images = batch["images"].to(DEVICE)
        true_idx = int(batch["label"][0].item())
        true_label = labels[true_idx]
        start_idx = int(batch["start"][0].item()) if "start" in batch else 0
        text_path = ""
        if "text_path" in batch and isinstance(batch["text_path"], list) and batch["text_path"]:
            text_path = str(batch["text_path"][0])

        gait_json_path = os.path.join(gait_json_dir, f"HSMR-{seq_id}.gait.json")
        gait_text = _load_gait_json_as_text(gait_json_path)
        skeleton_text = _load_skeleton_text(text_path=text_path, start=start_idx, window_size=WINDOW_SIZE)
        cls_prompt = _build_classification_prompt(skeleton_text)

        enc = tokenizer(
            [cls_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_TOKENS,
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        feats_img = _extract_image_text_feats(img_model, images, input_ids, attention_mask)
        feats_qm = _extract_qm_feats(decoder, images)
        feats = torch.cat([feats_img, feats_qm], dim=1)

        logits = classifier(feats)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(logits.argmax(dim=-1).item())
        pred_label = labels[pred_idx]
        pred_confidence = float(probs[0, pred_idx].item())

        topk = min(3, len(labels))
        top_vals, top_idxs = torch.topk(probs[0], k=topk, largest=True, sorted=True)
        top3_parts = []
        for v, i in zip(top_vals.tolist(), top_idxs.tolist()):
            top3_parts.append(f"{labels[int(i)]}:{float(v):.4f}")
        top3_text = ", ".join(top3_parts)

        is_correct = int(pred_idx == true_idx)
        is_correct_dcm = int(pred_label == "dcm")

        diag_prompt = _make_diagnosis_prompt(
            seq_id=seq_id,
            pred_label=pred_label,
            pred_confidence=pred_confidence,
            top3_text=top3_text,
            gait_text=gait_text,
            user_instruction=args.instruction,
        )
        diagnosis_text = _generate_diagnosis(
            base_model=base_model,
            tokenizer=tokenizer,
            images_batched=images,
            prompt=diag_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        diagnosis_class = _extract_final_diagnosis_class(diagnosis_text)
        diagnosis_matches_pred = int(diagnosis_class == pred_label.lower()) if diagnosis_class else 0
        diagnosis_matches_truth = int(diagnosis_class == true_label.lower()) if diagnosis_class else 0

        logit_dcm = float(logits[0, dcm_idx].item())
        prob_dcm = float(probs[0, dcm_idx].item())

        records.append(
            {
                "seq_id": seq_id,
                "true_label": true_label,
                "pred_label": pred_label,
                "is_correct_label": is_correct,
                "is_pred_dcm": is_correct_dcm,
                "logit_dcm": logit_dcm,
                "prob_dcm": prob_dcm,
                "pred_confidence": pred_confidence,
                "top3_probs": top3_text,
                "gait_json_path": gait_json_path,
                "text_path": text_path,
                "diagnosis_class_extracted": diagnosis_class,
                "diagnosis_matches_pred": diagnosis_matches_pred,
                "diagnosis_matches_truth": diagnosis_matches_truth,
                "diagnosis": diagnosis_text,
            }
        )

        correct += is_correct
        correct_dcm_binary += is_correct_dcm
        total += 1

    acc = correct / max(1, total)
    dcm_hit_rate = correct_dcm_binary / max(1, total)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False, encoding="utf-8")

    summary = {
        "split_csv": split_csv,
        "checkpoint": ckpt_path,
        "num_samples": total,
        "classification_accuracy": acc,
        "predicted_as_dcm_ratio": dcm_hit_rate,
        "all_gt_are_dcm_expected": True,
        "output_csv": output_csv,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-sample results to: {output_csv}")
    print(f"Saved summary to: {output_json}")
    print(f"Accuracy (exact label): {acc * 100:.2f}%")
    print(f"Predicted as dcm ratio: {dcm_hit_rate * 100:.2f}%")


if __name__ == "__main__":
    main()


