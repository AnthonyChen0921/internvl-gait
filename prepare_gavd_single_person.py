import os
import glob
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HSMR_DIR = os.path.join(BASE_DIR, "GAVD-HSMR")
HSMR_SINGLE_DIR = os.path.join(BASE_DIR, "GAVD-HSMR-single")
ANNOT_PATTERN = os.path.join(BASE_DIR, "GAVD", "data", "GAVD_Clinical_Annotations_*.csv")


def load_seq_to_label():
    """Load mapping from seq id -> gait_pat label from all CSV annotations."""
    csv_paths = sorted(glob.glob(ANNOT_PATTERN))
    if not csv_paths:
        raise FileNotFoundError(f"No annotation CSVs found matching {ANNOT_PATTERN}")

    dfs = []
    for p in csv_paths:
        print(f"Loading annotations from {p}")
        df = pd.read_csv(p, low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    # Some seqs have multiple rows; assume gait_pat is consistent and take first.
    grouped = df.groupby("seq")["gait_pat"].first()
    return grouped.to_dict()


def main():
    os.makedirs(HSMR_SINGLE_DIR, exist_ok=True)

    seq_to_label = load_seq_to_label()

    npy_paths = sorted(glob.glob(os.path.join(HSMR_DIR, "HSMR-*.npy")))
    print(f"Found {len(npy_paths)} HSMR .npy files in {HSMR_DIR}")

    kept_seqs = []
    dropped_seqs = []
    label_counter = Counter()
    unlabeled_seqs = []

    # For optional extra stats: how many frames per seq, etc.
    frames_per_seq = {}

    for path in npy_paths:
        fname = os.path.basename(path)
        # filenames look like: HSMR-<seq>.npy
        if not fname.startswith("HSMR-") or not fname.endswith(".npy"):
            continue
        seq_id = fname[len("HSMR-") : -len(".npy")]

        arr = np.load(path, allow_pickle=True)
        # arr is an object array of per-frame dicts with keys: poses, patch_cam_t, betas, bbx_cs
        multi_person = False
        single_person_poses = []

        for frame_dict in arr:
            poses = frame_dict["poses"]  # shape: (num_people, 46)
            num_people, feat_dim = poses.shape
            if num_people != 1:
                multi_person = True
                break
            # keep the single person's 46-dim vector
            single_person_poses.append(poses[0])

        if multi_person:
            dropped_seqs.append(seq_id)
            continue

        # All frames are single-person; stack into [T, 46] and save
        if not single_person_poses:
            # Empty sequence; drop
            dropped_seqs.append(seq_id)
            continue

        poses_arr = np.stack(single_person_poses, axis=0).astype(np.float32)  # [T, 46]
        out_path = os.path.join(HSMR_SINGLE_DIR, fname)
        np.save(out_path, poses_arr)

        kept_seqs.append(seq_id)
        frames_per_seq[seq_id] = poses_arr.shape[0]

        label = seq_to_label.get(seq_id)
        if isinstance(label, str) and len(label.strip()) > 0:
            label_counter[label.strip()] += 1
        else:
            unlabeled_seqs.append(seq_id)

    print("\n=== Filtering summary ===")
    print(f"Total sequences (npy files): {len(npy_paths)}")
    print(f"Kept single-person sequences: {len(kept_seqs)}")
    print(f"Dropped multi-person or empty sequences: {len(dropped_seqs)}")

    print("\n=== Label summary for kept sequences ===")
    if label_counter:
        for label, count in label_counter.most_common():
            print(f"  {label}: {count}")
    else:
        print("  No labels found for kept sequences.")

    if unlabeled_seqs:
        print(f"\nSequences without gait_pat label (kept): {len(unlabeled_seqs)}")

    # Optional: basic stats on frames per sequence
    if frames_per_seq:
        lengths = np.array(list(frames_per_seq.values()), dtype=np.int32)
        print("\n=== Frame count stats for kept sequences ===")
        print(f"  min frames: {lengths.min()}")
        print(f"  max frames: {lengths.max()}")
        print(f"  mean frames: {lengths.mean():.1f}")


if __name__ == "__main__":
    main()


