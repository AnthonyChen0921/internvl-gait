import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from internvl_skeleton_adapter import InternVLWithSkeleton
from internvl_temporal_adapter import EVLTemporalDecoder
from gavd_skeleton_dataset import GavdSkeletonDataset, TOP7_LABELS
from train_evl_dualstream_skeltext_classifier import (
    WINDOW_SIZE,
    BATCH_SIZE,
    _extract_image_text_feats,
    _extract_qm_feats,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPLIT_CSV = os.path.join(
    BASE_DIR,
    "splits",
    "gavd_plus_dcm_singlelabel_legacygavd",
    "dcm_train.csv",
)
DEFAULT_CKPT = os.path.join(BASE_DIR, "best_evl_dualstream_skeltext_classifier.pt")
DEFAULT_GAIT_PARAM_DIR = os.path.join(BASE_DIR, "gait_parameter_results")

ALL_LABELS = TOP7_LABELS + ["dcm"]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(ALL_LABELS)}


def _resolve(p: str) -> str:
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(BASE_DIR, p))


def _load_samples_from_split_csv(csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    required = {"seq_id", "label", "skeleton_path", "video_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV missing columns {sorted(missing)}: {csv_path}")

    has_text = "text_path" in df.columns
    samples: List[Dict] = []
    for _, row in df.iterrows():
        label = str(row["label"])
        if label not in LABEL_TO_IDX:
            raise ValueError(f"Unknown label {label!r} in {csv_path}")
        sample = {
            "seq_id": str(row["seq_id"]),
            "path": _resolve(str(row["skeleton_path"])),
            "video_path": _resolve(str(row["video_path"])),
            "label_str": label,
            "label_idx": int(LABEL_TO_IDX[label]),
        }
        if has_text and isinstance(row["text_path"], str) and row["text_path"].strip():
            sample["text_path"] = _resolve(row["text_path"])
        samples.append(sample)
    return samples


def _load_gait_json_summary(seq_id: str, gait_param_dir: str) -> str:
    gait_json = os.path.join(gait_param_dir, f"HSMR-{seq_id}.gait.json")
    if not os.path.exists(gait_json):
        return "No gait-parameter JSON file found for this sequence."

    try:
        with open(gait_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:  # noqa: BLE001
        return f"Failed to read gait-parameter JSON: {e}"

    events = obj.get("events", {})
    counts = events.get("counts", {})
    metrics = obj.get("metrics", {})

    keep_metrics = [
        "steps_per_minute",
        "left_step_time",
        "right_step_time",
        "left_stride_time",
        "right_stride_time",
        "left_swing_percent",
        "right_swing_percent",
    ]
    metric_parts = []
    for k in keep_metrics:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                metric_parts.append(f"{k}={v:.4f}")
            else:
                metric_parts.append(f"{k}={v}")

    count_parts = []
    for k in ["HS_L", "HS_R", "TO_L_valid", "TO_R_valid"]:
        if k in counts:
            count_parts.append(f"{k}={counts[k]}")

    return (
        f"num_frames={obj.get('num_frames', 'NA')}, fps={obj.get('fps', 'NA')}; "
        f"event_counts: {', '.join(count_parts) if count_parts else 'NA'}; "
        f"metrics: {', '.join(metric_parts) if metric_parts else 'NA'}"
    )


def _build_classification_prompt(batch_size: int, tokenizer, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
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
        "- dcm: degenerative cervical myelopathy-related gait impairment.\n\n"
        "Decide the most likely class internally."
    )
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device).expand(batch_size, -1).contiguous()
    attention_mask = enc["attention_mask"].to(device).expand(batch_size, -1).contiguous()
    return input_ids, attention_mask


def _generate_diagnosis(
    tokenizer,
    language_model,
    decoder: EVLTemporalDecoder,
    images: torch.Tensor,
    seq_id: str,
    predicted_label: str,
    gait_summary: str,
    device: str,
    max_new_tokens: int,
) -> str:
    instruction = (
        "You are an expert gait clinician.\n"
        f"Sequence ID: {seq_id}\n"
        f"Classifier predicted label: {predicted_label}\n"
        f"Precomputed gait parameters: {gait_summary}\n\n"
        "Provide a concise diagnosis in 2-4 sentences. Mention likely gait abnormalities and why "
        "the observed gait could be consistent or inconsistent with DCM."
    )
    enc = tokenizer(instruction, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        video_tokens = decoder(pixel_values=images).to(dtype=language_model.get_input_embeddings().weight.dtype)
        B, T, _ = video_tokens.shape
        text_embeds = language_model.get_input_embeddings()(input_ids)
        video_mask = torch.ones(B, T, dtype=attention_mask.dtype, device=device)
        inputs_embeds = torch.cat([video_tokens, text_embeds], dim=1)
        fused_mask = torch.cat([video_mask, attention_mask], dim=1)

        outputs = language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=fused_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=str, default=DEFAULT_SPLIT_CSV, help="CSV split to evaluate.")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Checkpoint path.")
    parser.add_argument(
        "--gait-parameter-dir",
        type=str,
        default=DEFAULT_GAIT_PARAM_DIR,
        help="Directory containing HSMR-<seq_id>.gait.json",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(BASE_DIR, "reports", "dcm_dualstream_skeltext_diagnosis_results.csv"),
        help="Output CSV with per-sample predictions and diagnosis text.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--diagnosis-limit",
        type=int,
        default=20,
        help="Generate diagnosis text only for the first N samples. Use -1 for all, 0 for none.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, evaluate only first N samples (for quick sanity checks).",
    )
    args = parser.parse_args()

    split_csv = _resolve(args.split_csv)
    ckpt_path = _resolve(args.ckpt)
    gait_dir = _resolve(args.gait_parameter_dir)
    out_csv = _resolve(args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    samples = _load_samples_from_split_csv(split_csv)
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} samples from {split_csv}")

    ds = GavdSkeletonDataset(
        samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=BASE_DIR,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    tokenizer, base_model, _ = load_model(device=DEVICE)
    base_model.requires_grad_(False)
    language_model = getattr(base_model, "language_model", base_model)
    language_model.eval()

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
    classifier = nn.Linear(hidden_size * 2, len(ALL_LABELS), dtype=torch.float32).to(DEVICE)
    classifier.eval()

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    decoder.load_state_dict(ckpt["decoder"])
    classifier.load_state_dict(ckpt["classifier"])
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Checkpoint epoch={ckpt.get('epoch', 'NA')} macro_f1={ckpt.get('macro_f1', 'NA')}")

    trained_classes = int(classifier.weight.shape[0])
    if trained_classes != len(ALL_LABELS):
        raise ValueError(
            f"Checkpoint classifier output size is {trained_classes}, expected {len(ALL_LABELS)}. "
            "This script requires an 8-class model (TOP7 + dcm)."
        )

    dcm_idx = LABEL_TO_IDX["dcm"]
    correct = 0
    total = 0
    rows: List[Dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating DCM set"):
            images = batch["images"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            seq_ids = batch["seq_id"]

            input_ids, attention_mask = _build_classification_prompt(images.size(0), tokenizer, DEVICE)
            feats_img = _extract_image_text_feats(img_model, images, input_ids, attention_mask)
            feats_qm = _extract_qm_feats(decoder, images)
            feats = torch.cat([feats_img, feats_qm], dim=1)
            logits = classifier(feats)
            preds = logits.argmax(dim=-1)

            for i in range(images.size(0)):
                seq_id = str(seq_ids[i])
                true_idx = int(labels[i].item())
                pred_idx = int(preds[i].item())
                is_correct = int(pred_idx == true_idx)
                is_pred_dcm = int(pred_idx == dcm_idx)

                correct += is_correct
                total += 1

                gait_summary = _load_gait_json_summary(seq_id, gait_dir)
                should_generate = args.diagnosis_limit < 0 or len(rows) < args.diagnosis_limit
                if should_generate:
                    diagnosis = _generate_diagnosis(
                        tokenizer=tokenizer,
                        language_model=language_model,
                        decoder=decoder,
                        images=images[i : i + 1],
                        seq_id=seq_id,
                        predicted_label=ALL_LABELS[pred_idx],
                        gait_summary=gait_summary,
                        device=DEVICE,
                        max_new_tokens=args.max_new_tokens,
                    )
                else:
                    diagnosis = ""

                rows.append(
                    {
                        "seq_id": seq_id,
                        "true_label": ALL_LABELS[true_idx],
                        "pred_label": ALL_LABELS[pred_idx],
                        "is_pred_correct": is_correct,
                        "is_pred_dcm": is_pred_dcm,
                        "gait_summary": gait_summary,
                        "diagnosis_text": diagnosis,
                    }
                )

    acc = correct / max(1, total)
    dcm_pred = sum(r["is_pred_dcm"] for r in rows)
    print(f"\nTotal samples: {total}")
    print(f"Classification accuracy (on this CSV): {acc * 100:.2f}%")
    print(f"Predicted as dcm: {dcm_pred}/{total}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seq_id",
                "true_label",
                "pred_label",
                "is_pred_correct",
                "is_pred_dcm",
                "gait_summary",
                "diagnosis_text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved detailed results: {out_csv}")
    print("\nSample outputs:")
    for r in rows[:3]:
        print("-" * 80)
        print(f"seq_id={r['seq_id']} true={r['true_label']} pred={r['pred_label']} correct={r['is_pred_correct']}")
        print(f"diagnosis: {r['diagnosis_text'][:400]}")


if __name__ == "__main__":
    main()


