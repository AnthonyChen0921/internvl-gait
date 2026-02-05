import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gait_videos_name_stats import parse_stem


TOP7_LABELS = [
    "abnormal",
    "myopathic",
    "exercise",
    "normal",
    "style",
    "cerebral palsy",
    "parkinsons",
]


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def get_video_meta(path: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Returns (frame_count, fps) from OpenCV metadata.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS))
    fc = _safe_float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_count = int(fc) if fc is not None and fc > 0 else None
    fps_val = fps if fps is not None and fps > 0 else None
    return frame_count, fps_val


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
        description="Compare GAVD frame-length distributions vs your clipped gait-videos (by type)."
    )
    parser.add_argument(
        "--gavd-csv",
        type=str,
        default="gavd_exp_seq_lengths.csv",
        help="CSV produced earlier with columns: seq_id,video_id,label,num_frames",
    )
    parser.add_argument(
        "--clipped-dir",
        type=str,
        default="Gait-videos-clipped",
        help="Directory containing clipped videos (*.mp4).",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="gavd_vs_clipped_frame_distribution.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--out-clipped-csv",
        type=str,
        default="clipped_video_frame_lengths.csv",
        help="Write per-video frame counts for clipped videos to this CSV.",
    )
    parser.add_argument(
        "--exclude-type",
        action="append",
        default=[],
        help=(
            "Exclude clipped test type(s) from the chart/CSVs (can be specified multiple times). "
            "Example: --exclude-type FwdWalkData"
        ),
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    gavd_csv = args.gavd_csv if os.path.isabs(args.gavd_csv) else os.path.join(base, args.gavd_csv)
    clipped_dir = args.clipped_dir if os.path.isabs(args.clipped_dir) else os.path.join(base, args.clipped_dir)
    out_png = args.out_png if os.path.isabs(args.out_png) else os.path.join(base, args.out_png)
    out_clipped_csv = (
        args.out_clipped_csv if os.path.isabs(args.out_clipped_csv) else os.path.join(base, args.out_clipped_csv)
    )

    if not os.path.isfile(gavd_csv):
        raise FileNotFoundError(f"GAVD lengths CSV not found: {gavd_csv}")
    if not os.path.isdir(clipped_dir):
        raise FileNotFoundError(f"Clipped dir not found: {clipped_dir}")

    # ---- Load GAVD lengths ----
    gavd = pd.read_csv(gavd_csv)
    gavd["label"] = gavd["label"].astype(str)
    gavd["num_frames"] = pd.to_numeric(gavd["num_frames"], errors="coerce")
    gavd = gavd.dropna(subset=["num_frames"])
    gavd["num_frames"] = gavd["num_frames"].astype(int)

    # Ensure only TOP7 and consistent order
    gavd = gavd[gavd["label"].isin(TOP7_LABELS)].copy()
    gavd["label"] = pd.Categorical(gavd["label"], categories=TOP7_LABELS, ordered=True)

    gavd_groups = [gavd.loc[gavd["label"] == lbl, "num_frames"].to_numpy() for lbl in TOP7_LABELS]
    gavd_counts = [int((gavd["label"] == lbl).sum()) for lbl in TOP7_LABELS]
    gavd_totals = [int(g.sum()) if len(g) else 0 for g in gavd_groups]

    # ---- Scan clipped videos and compute frame counts ----
    clipped_rows: List[Dict[str, object]] = []
    unparsed: List[str] = []
    meta_missing: List[str] = []

    for fn in sorted(os.listdir(clipped_dir)):
        path = os.path.join(clipped_dir, fn)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(fn)[1].lower() != ".mp4":
            continue

        stem = os.path.splitext(fn)[0]
        # strip suffix used by our clipper
        if stem.endswith("_clip"):
            stem = stem[: -len("_clip")]
        parsed = parse_stem(stem)
        if parsed is None:
            unparsed.append(fn)
            continue

        frame_count, fps = get_video_meta(path)
        if frame_count is None:
            meta_missing.append(fn)
            continue

        clipped_rows.append(
            {
                "path": path,
                "filename": fn,
                "cohort": parsed.cohort,
                "subject_id": parsed.subject_id,
                "type": parsed.test_type,
                "run": parsed.run,
                "frame_count": int(frame_count),
                "fps": float(fps) if fps is not None else np.nan,
                "duration_sec": (float(frame_count) / float(fps)) if fps is not None and fps > 0 else np.nan,
            }
        )

    if not clipped_rows:
        raise RuntimeError(
            "No clipped videos could be parsed with frame counts. "
            "Make sure your clipped files look like DCM_<id>_<Type>_<run>_clip.mp4"
        )

    clipped_df = pd.DataFrame(clipped_rows)

    # Optionally exclude specific types
    exclude_types = {str(t).strip() for t in args.exclude_type if str(t).strip()}
    if exclude_types:
        clipped_df = clipped_df[~clipped_df["type"].isin(exclude_types)].copy()

    clipped_df.to_csv(out_clipped_csv, index=False, encoding="utf-8")

    type_order = sorted(clipped_df["type"].unique().tolist())
    if not type_order:
        raise RuntimeError(f"After excluding types {sorted(exclude_types)}, no clipped types remain to plot.")
    clip_groups = [clipped_df.loc[clipped_df["type"] == t, "frame_count"].to_numpy() for t in type_order]
    clip_counts = [int((clipped_df["type"] == t).sum()) for t in type_order]
    clip_totals = [int(g.sum()) if len(g) else 0 for g in clip_groups]

    # ---- Plot ----
    fig = plt.figure(figsize=(16, 9), dpi=160)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.3], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    boxplot(
        ax1,
        gavd_groups,
        TOP7_LABELS,
        title="GAVD experiment subset: per-sequence skeleton frame length [T,46] (whiskers 5–95%)",
        ylabel="Frames per sequence (T)",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    barplot(
        ax2,
        TOP7_LABELS,
        gavd_totals,
        gavd_counts,
        title="GAVD total frames by class (sum over sequences)",
        ylabel="Total frames",
    )

    ax3 = fig.add_subplot(gs[1, 0])
    boxplot(
        ax3,
        clip_groups,
        type_order,
        title="Clipped gait-videos: per-video frame count by test type (whiskers 5–95%)",
        ylabel="Frames per clipped video",
    )

    ax4 = fig.add_subplot(gs[1, 1])
    barplot(
        ax4,
        type_order,
        clip_totals,
        clip_counts,
        title="Clipped gait-videos total frames by type (sum over videos)",
        ylabel="Total frames",
    )

    # Align y-limits on the two boxplots so comparisons are visual.
    ymax = max(
        [float(np.max(g)) for g in gavd_groups if len(g)] + [float(np.max(g)) for g in clip_groups if len(g)]
    )
    for ax in (ax1, ax3):
        ax.set_ylim(0, ymax * 1.05)

    fig.suptitle(
        "Frame-count comparison: GAVD skeleton sequences vs your clipped gait videos\n"
        "(Note: GAVD counts are skeleton sequence length; clipped counts are video frames.)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Comparison exported ===")
    print(f"GAVD sequences: {len(gavd)}  (classes: {len(TOP7_LABELS)})")
    print(f"Clipped videos: {len(clipped_df)}  (types: {len(type_order)})")
    print(f"Wrote plot: {out_png}")
    print(f"Wrote clipped per-video CSV: {out_clipped_csv}")
    if unparsed:
        print(f"Unparsed clipped filenames: {len(unparsed)} (first 10: {unparsed[:10]})")
    if meta_missing:
        print(f"Clipped videos missing frame_count metadata: {len(meta_missing)} (first 10: {meta_missing[:10]})")


if __name__ == "__main__":
    main()



