import argparse
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOP7_LABELS = [
    "abnormal",
    "myopathic",
    "exercise",
    "normal",
    "style",
    "cerebral palsy",
    "parkinsons",
]


def boxplot(ax, groups: List[np.ndarray], labels: List[str], title: str, ylabel: str):
    ax.boxplot(groups, tick_labels=labels, showfliers=False, whis=(5, 95))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)


def barplot(ax, labels: List[str], totals: List[int], counts: List[int], title: str, ylabel: str):
    x = np.arange(len(labels))
    bars = ax.bar(x, totals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    for b, n in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={n}", ha="center", va="bottom", fontsize=8)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a merged chart that appends DCM clipped-video categories to the GAVD frame distribution, "
            "as if they were part of the same dataset."
        )
    )
    parser.add_argument(
        "--gavd-csv",
        type=str,
        default="gavd_exp_seq_lengths.csv",
        help="CSV with columns: seq_id,video_id,label,num_frames (from gavd_data_stats.py).",
    )
    parser.add_argument(
        "--dcm-csv",
        type=str,
        default="clipped_video_frame_lengths.csv",
        help="CSV with columns including: type,frame_count (from compare_gavd_vs_clipped_lengths.py).",
    )
    parser.add_argument(
        "--exclude-dcm-type",
        action="append",
        default=[],
        help="Exclude DCM type(s) (can be specified multiple times). Example: --exclude-dcm-type FwdWalkData",
    )
    parser.add_argument(
        "--combine-walk-fast",
        action="store_true",
        help="Combine DCM WalkData and FastWalkData into a single pooled category.",
    )
    parser.add_argument(
        "--combined-label",
        type=str,
        default="dcm",
        help="Label to use when combining WalkData+FastWalkData (default: DCM:Walk+Fast).",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="gavd_plus_dcm_frame_distribution.png",
        help="Output merged chart PNG path.",
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    gavd_csv = args.gavd_csv if os.path.isabs(args.gavd_csv) else os.path.join(base, args.gavd_csv)
    dcm_csv = args.dcm_csv if os.path.isabs(args.dcm_csv) else os.path.join(base, args.dcm_csv)
    out_png = args.out_png if os.path.isabs(args.out_png) else os.path.join(base, args.out_png)

    if not os.path.isfile(gavd_csv):
        raise FileNotFoundError(f"GAVD CSV not found: {gavd_csv}")
    if not os.path.isfile(dcm_csv):
        raise FileNotFoundError(f"DCM clipped CSV not found: {dcm_csv}")

    # ---- Load GAVD lengths (skeleton sequence lengths) ----
    gavd = pd.read_csv(gavd_csv)
    gavd["label"] = gavd["label"].astype(str)
    gavd["num_frames"] = pd.to_numeric(gavd["num_frames"], errors="coerce")
    gavd = gavd.dropna(subset=["num_frames"])
    gavd["num_frames"] = gavd["num_frames"].astype(int)
    gavd = gavd[gavd["label"].isin(TOP7_LABELS)].copy()

    # ---- Load DCM clipped lengths (video frame counts) ----
    dcm = pd.read_csv(dcm_csv)
    if "type" not in dcm.columns or "frame_count" not in dcm.columns:
        raise ValueError(f"DCM CSV must contain 'type' and 'frame_count' columns: {dcm_csv}")
    dcm["type"] = dcm["type"].astype(str)
    dcm["frame_count"] = pd.to_numeric(dcm["frame_count"], errors="coerce")
    dcm = dcm.dropna(subset=["frame_count"]).copy()
    dcm["frame_count"] = dcm["frame_count"].astype(int)

    exclude_types = {str(t).strip() for t in args.exclude_dcm_type if str(t).strip()}
    if exclude_types:
        dcm = dcm[~dcm["type"].isin(exclude_types)].copy()

    # Categories: TOP7 + DCM categories (optionally combining WalkData+FastWalkData)
    dcm_types_all = sorted(dcm["type"].unique().tolist())
    dcm_types_to_plot: List[str] = []
    combined_types = {"WalkData", "FastWalkData"}
    do_combine = bool(args.combine_walk_fast) and any(t in combined_types for t in dcm_types_all)

    if do_combine:
        # Represent the combined pool by a sentinel type key
        dcm_types_to_plot.append("__COMBINED_WALK_FAST__")
        # Keep any other DCM types separate
        for t in dcm_types_all:
            if t not in combined_types:
                dcm_types_to_plot.append(t)
    else:
        dcm_types_to_plot = dcm_types_all[:]

    def dcm_display_label(t: str) -> str:
        if t == "__COMBINED_WALK_FAST__":
            return str(args.combined_label)
        return f"DCM:{t}"

    dcm_labels = [dcm_display_label(t) for t in dcm_types_to_plot]
    all_labels = TOP7_LABELS + dcm_labels

    groups: List[np.ndarray] = []
    counts: List[int] = []
    totals: List[int] = []

    # GAVD groups
    for lbl in TOP7_LABELS:
        arr = gavd.loc[gavd["label"] == lbl, "num_frames"].to_numpy(dtype=np.int32)
        groups.append(arr)
        counts.append(int(arr.size))
        totals.append(int(arr.sum()) if arr.size else 0)

    # DCM groups
    for t in dcm_types_to_plot:
        if t == "__COMBINED_WALK_FAST__":
            arr = dcm.loc[dcm["type"].isin(list(combined_types)), "frame_count"].to_numpy(dtype=np.int32)
        else:
            arr = dcm.loc[dcm["type"] == t, "frame_count"].to_numpy(dtype=np.int32)
        groups.append(arr)
        counts.append(int(arr.size))
        totals.append(int(arr.sum()) if arr.size else 0)

    # ---- Plot ----
    fig = plt.figure(figsize=(16, 5), dpi=160)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.3])

    ax1 = fig.add_subplot(gs[0, 0])
    boxplot(
        ax1,
        groups,
        all_labels,
        title="Merged (pretend): GAVD TOP7 + DCM clipped types — per-item frame lengths (whiskers 5–95%)",
        ylabel="Frames (GAVD: skeleton T, DCM: video frames)",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    barplot(
        ax2,
        all_labels,
        totals,
        counts,
        title="Merged (pretend): total frames per category (sum over items)",
        ylabel="Total frames",
    )

    # Shared y-range helps eyeballing scale differences
    ymax = 0.0
    for g in groups:
        if g.size:
            ymax = max(ymax, float(np.max(g)))
    if ymax > 0:
        ax1.set_ylim(0, ymax * 1.05)

    fig.suptitle(
        "NOTE: This is a *synthetic* merge for comparison only.\n"
        "GAVD uses skeleton sequence length T; DCM uses video frame_count from .mp4 metadata.",
        y=1.05,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Merged chart exported ===")
    print(f"GAVD items: {len(gavd)} (TOP7)")
    print(f"DCM clipped items: {len(dcm)} (types={len(dcm_types_all)})")
    if exclude_types:
        print(f"Excluded DCM types: {sorted(exclude_types)}")
    if do_combine:
        print(f"Combined DCM types into '{args.combined_label}': {sorted(combined_types)}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()


