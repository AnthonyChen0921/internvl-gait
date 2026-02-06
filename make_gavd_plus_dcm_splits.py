import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _as_rel_posix(*parts: str) -> str:
    # Create repo-relative paths with forward slashes for portability (Windows/Linux).
    return "/".join([p.strip("/\\") for p in parts if p is not None and str(p) != ""])


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    train_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split over groups using a single label per group.
    """
    rng = random.Random(seed)
    group_to_label: Dict[str, str] = {}
    for gid, g in df.groupby(group_col):
        # assume consistent; take first
        group_to_label[str(gid)] = str(g[label_col].iloc[0])

    by_label: Dict[str, List[str]] = defaultdict(list)
    for gid, lbl in group_to_label.items():
        by_label[lbl].append(gid)

    train_groups = set()
    test_groups = set()
    for lbl, groups in by_label.items():
        groups = groups[:]
        rng.shuffle(groups)
        n = len(groups)
        n_train = max(1, int(round(train_ratio * n)))
        if n > 1 and n_train == n:
            n_train = n - 1
        train_groups.update(groups[:n_train])
        test_groups.update(groups[n_train:])

    train_df = df[df[group_col].astype(str).isin(train_groups)].copy()
    test_df = df[df[group_col].astype(str).isin(test_groups)].copy()
    return train_df, test_df


def group_split(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random split over groups (no stratification).
    """
    rng = random.Random(seed)
    groups = sorted(df[group_col].astype(str).unique().tolist())
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, int(round(train_ratio * n)))
    if n > 1 and n_train == n:
        n_train = n - 1
    train_groups = set(groups[:n_train])
    test_groups = set(groups[n_train:])
    return (
        df[df[group_col].astype(str).isin(train_groups)].copy(),
        df[df[group_col].astype(str).isin(test_groups)].copy(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create train/test splits for GAVD + DCM with correct leakage rules."
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    # Inputs (already produced earlier in this repo)
    parser.add_argument("--gavd-csv", type=str, default="reports/gavd/gavd_exp_seq_lengths.csv")
    parser.add_argument("--dcm-csv", type=str, default="reports/dcm/dcm_track_audit_filtered.csv")

    # DCM label choice
    parser.add_argument(
        "--dcm-label-mode",
        choices=["type", "single"],
        default="type",
        help="How to label DCM samples: 'type' => dcm:WalkData/dcm:FastWalkData..., 'single' => dcm",
    )

    # Output
    parser.add_argument("--out-dir", type=str, default="splits/gavd_plus_dcm")
    args = parser.parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(BASE_DIR, args.out_dir)
    _ensure_dir(out_dir)

    gavd_csv = args.gavd_csv if os.path.isabs(args.gavd_csv) else os.path.join(BASE_DIR, args.gavd_csv)
    dcm_csv = args.dcm_csv if os.path.isabs(args.dcm_csv) else os.path.join(BASE_DIR, args.dcm_csv)

    # ---- Load GAVD samples (video-level split constraint) ----
    gavd = pd.read_csv(gavd_csv)
    required_gavd = {"seq_id", "video_id", "label"}
    if not required_gavd.issubset(set(gavd.columns)):
        raise ValueError(f"GAVD CSV must contain columns {sorted(required_gavd)}: {gavd_csv}")

    gavd["dataset"] = "gavd"
    gavd["group_id"] = gavd["video_id"].astype(str)
    gavd["label"] = gavd["label"].astype(str)
    gavd["seq_id"] = gavd["seq_id"].astype(str)

    # Write repo-relative paths (portable)
    gavd["skeleton_path"] = gavd["seq_id"].map(lambda sid: _as_rel_posix("GAVD-HSMR-single", f"HSMR-{sid}.npy"))
    gavd["video_path"] = gavd["seq_id"].map(lambda sid: _as_rel_posix("GAVD-sequences", f"{sid}.mp4"))
    gavd["text_path"] = gavd["seq_id"].map(lambda sid: _as_rel_posix("GAVD-HSMR-text", f"HSMR-{sid}.jsonl"))

    # ---- Load DCM samples (subject-level split constraint) ----
    dcm = pd.read_csv(dcm_csv)
    required_dcm = {"seq_id", "subject_id", "type", "T_out"}
    if not required_dcm.issubset(set(dcm.columns)):
        raise ValueError(f"DCM CSV must contain columns {sorted(required_dcm)}: {dcm_csv}")

    dcm["dataset"] = "dcm"
    dcm["seq_id"] = dcm["seq_id"].astype(str)
    dcm["group_id"] = dcm["subject_id"].astype(str)  # patient-level leakage rule

    if args.dcm_label_mode == "single":
        dcm["label"] = "dcm"
    else:
        dcm["label"] = dcm["type"].astype(str).map(lambda t: f"dcm:{t}")

    # IMPORTANT: Use the filtered oneperson folders we created
    dcm["skeleton_path"] = dcm["seq_id"].map(
        lambda sid: _as_rel_posix("GAVD-HSMR-dcm-oneperson-filtered", f"HSMR-{sid}.npy")
    )
    dcm["video_path"] = dcm["seq_id"].map(
        lambda sid: _as_rel_posix("GAVD-videos-clipped-oneperson-filtered", f"{sid}.mp4")
    )
    dcm["text_path"] = dcm["seq_id"].map(
        lambda sid: _as_rel_posix("GAVD-HSMR-dcm-text-oneperson-filtered", f"HSMR-{sid}.jsonl")
    )

    # ---- Split ----
    # GAVD: stratify over videos by label (matches your previous rule)
    gavd_train, gavd_test = stratified_group_split(
        gavd, group_col="group_id", label_col="label", train_ratio=args.train_ratio, seed=args.seed
    )
    # DCM: split over subjects (patients). Stratification is non-trivial because a subject has multiple types.
    dcm_train, dcm_test = group_split(dcm, group_col="group_id", train_ratio=args.train_ratio, seed=args.seed)

    # Leak checks
    assert len(set(gavd_train["group_id"]) & set(gavd_test["group_id"])) == 0
    assert len(set(dcm_train["group_id"]) & set(dcm_test["group_id"])) == 0

    combined_train = pd.concat([gavd_train, dcm_train], ignore_index=True)
    combined_test = pd.concat([gavd_test, dcm_test], ignore_index=True)

    # Save CSVs
    cols = ["dataset", "seq_id", "group_id", "label", "skeleton_path", "video_path", "text_path"]
    gavd_train[cols].to_csv(os.path.join(out_dir, "gavd_train.csv"), index=False, encoding="utf-8")
    gavd_test[cols].to_csv(os.path.join(out_dir, "gavd_test.csv"), index=False, encoding="utf-8")
    dcm_train[cols].to_csv(os.path.join(out_dir, "dcm_train.csv"), index=False, encoding="utf-8")
    dcm_test[cols].to_csv(os.path.join(out_dir, "dcm_test.csv"), index=False, encoding="utf-8")
    combined_train[cols].to_csv(os.path.join(out_dir, "combined_train.csv"), index=False, encoding="utf-8")
    combined_test[cols].to_csv(os.path.join(out_dir, "combined_test.csv"), index=False, encoding="utf-8")

    # Summary
    def _summ(df: pd.DataFrame) -> Dict[str, object]:
        return {
            "rows": int(len(df)),
            "groups": int(df["group_id"].nunique()),
            "labels": dict(Counter(df["label"].astype(str)).most_common()),
        }

    summary = {
        "params": {"train_ratio": args.train_ratio, "seed": args.seed, "dcm_label_mode": args.dcm_label_mode},
        "gavd": {"train": _summ(gavd_train), "test": _summ(gavd_test)},
        "dcm": {"train": _summ(dcm_train), "test": _summ(dcm_test)},
        "combined": {"train": _summ(combined_train), "test": _summ(combined_test)},
        "leak_checks": {
            "gavd_video_overlap_train_test": int(len(set(gavd_train["group_id"]) & set(gavd_test["group_id"]))),
            "dcm_subject_overlap_train_test": int(len(set(dcm_train["group_id"]) & set(dcm_test["group_id"]))),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Wrote splits ===")
    print(f"Out dir: {out_dir}")
    print(f"GAVD: train={len(gavd_train)} (videos={gavd_train['group_id'].nunique()}) | test={len(gavd_test)} (videos={gavd_test['group_id'].nunique()})")
    print(f"DCM:  train={len(dcm_train)} (subjects={dcm_train['group_id'].nunique()}) | test={len(dcm_test)} (subjects={dcm_test['group_id'].nunique()})")
    print(f"Combined: train={len(combined_train)} | test={len(combined_test)}")
    print("Leak checks OK.")


if __name__ == "__main__":
    main()


