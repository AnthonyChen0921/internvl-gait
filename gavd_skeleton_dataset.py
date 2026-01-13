import os
import glob
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HSMR_SINGLE_DIR = os.path.join(BASE_DIR, "GAVD-HSMR-single")
ANNOT_PATTERN = os.path.join(BASE_DIR, "GAVD", "data", "GAVD_Clinical_Annotations_*.csv")


# Keep 7 main gait pattern classes
TOP7_LABELS = [
    "abnormal",
    "myopathic",
    "exercise",
    "normal",
    "style",
    "cerebral palsy",
    "parkinsons",
]
LABEL_TO_IDX: Dict[str, int] = {lbl: i for i, lbl in enumerate(TOP7_LABELS)}


def load_seq_to_label_and_video() -> Dict[str, Dict[str, str]]:
    """
    Load mapping from seq id -> {'gait_pat': label_str, 'video_id': vid}
    from all CSV annotations.
    """
    csv_paths = sorted(glob.glob(ANNOT_PATTERN))
    if not csv_paths:
        raise FileNotFoundError(f"No annotation CSVs found matching {ANNOT_PATTERN}")

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p, low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # For each seq, take first gait_pat and first video id (`id` column)
    grouped = df.groupby("seq").agg({"gait_pat": "first", "id": "first"})
    seq_map: Dict[str, Dict[str, str]] = {}
    for seq_id, row in grouped.iterrows():
        seq_map[str(seq_id)] = {
            "gait_pat": str(row["gait_pat"]) if not pd.isna(row["gait_pat"]) else "",
            "video_id": str(row["id"]) if not pd.isna(row["id"]) else "",
        }
    return seq_map


def collect_labeled_sequences() -> List[Dict]:
    """
    Collect all single-person sequences that have one of the TOP7_LABELS.

    Returns:
        List of dicts: { 'seq_id', 'path', 'label_str', 'label_idx', 'num_frames' }
    """
    seq_info = load_seq_to_label_and_video()

    npy_paths = sorted(glob.glob(os.path.join(HSMR_SINGLE_DIR, "HSMR-*.npy")))
    if not npy_paths:
        raise FileNotFoundError(f"No single-person npy files found in {HSMR_SINGLE_DIR}")

    samples = []
    for path in npy_paths:
        fname = os.path.basename(path)
        seq_id = fname[len("HSMR-") : -len(".npy")]
        info = seq_info.get(seq_id)
        if info is None:
            continue
        raw_label = info.get("gait_pat", "")
        video_id = info.get("video_id", "")
        if not isinstance(raw_label, str) or not isinstance(video_id, str) or video_id == "":
            continue
        label_str = raw_label.strip()
        if label_str not in LABEL_TO_IDX:
            continue

        # Robustly load npy; skip corrupted/empty ones
        try:
            arr = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
            continue

        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] != 46:
            # Expect non-empty [T, 46] from prepare_gavd_single_person
            print(f"Warning: skipping malformed skeleton array {path} with shape {getattr(arr, 'shape', None)}")
            continue

        samples.append(
            {
                "seq_id": seq_id,
                "path": path,
                "label_str": label_str,
                "label_idx": LABEL_TO_IDX[label_str],
                "video_id": video_id,
                "num_frames": arr.shape[0],
            }
        )

    # Basic label distribution sanity check
    counter = Counter(s["label_str"] for s in samples)
    print("Label distribution among TOP7 samples:")
    for lbl in TOP7_LABELS:
        print(f"  {lbl}: {counter.get(lbl, 0)}")

    return samples


def video_level_train_test_split(
    samples: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split at the video level so that the same video_id never appears in both train and test.

    Strategy:
      - Group sequences by video_id.
      - Assign each video a label_idx (from the first sequence belonging to it).
      - Do a stratified split over videos by label_idx.
      - Expand back to sequences: all sequences from a video go to the same split.
    """
    rng = random.Random(seed)

    # Group sequences by video
    by_video: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        vid = s["video_id"]
        by_video[vid].append(s)

    # Assign a single label per video (assume all seqs from same vid share label)
    video_entries = []
    for vid, seqs in by_video.items():
        label_idx = seqs[0]["label_idx"]
        video_entries.append({"video_id": vid, "label_idx": label_idx})

    # Stratified split at video level
    by_label_vid: Dict[int, List[str]] = defaultdict(list)
    for e in video_entries:
        by_label_vid[e["label_idx"]].append(e["video_id"])

    train_videos = set()
    test_videos = set()
    for label_idx, vids in by_label_vid.items():
        vids = vids[:]  # copy
        rng.shuffle(vids)
        n = len(vids)
        n_train = max(1, int(round(train_ratio * n)))
        if n > 1 and n_train == n:
            n_train = n - 1
        train_videos.update(vids[:n_train])
        test_videos.update(vids[n_train:])

    # Map videos back to sequence samples
    train = [s for s in samples if s["video_id"] in train_videos]
    test = [s for s in samples if s["video_id"] in test_videos]

    return train, test


class GavdSkeletonDataset(Dataset):
    """
    Dataset over preprocessed single-person gait sequences in GAVD-HSMR-single.

    By default, each item returns:
      - skeleton: [T_window, 46] tensor
      - label: int in [0, 6]

    It can optionally also return aligned image frames when with_images=True.
    """

    def __init__(
        self,
        samples: List[Dict],
        window_size: int = 64,
        train: bool = True,
        with_images: bool = False,
        video_dir: Optional[str] = None,
        image_transform: Optional[object] = None,
    ):
        self.samples = samples
        self.window_size = window_size
        self.train = train
        self.with_images = with_images
        self.video_dir = video_dir
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_or_pad(self, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        arr: [T, 46]
        Returns:
          - window: [window_size, 46]
          - start index in the original sequence (for alignment with video frames)
        """
        T = arr.shape[0]
        W = self.window_size

        if T == W:
            return arr, 0
        if T < W:
            pad = np.zeros((W - T, arr.shape[1]), dtype=arr.dtype)
            return np.concatenate([arr, pad], axis=0), 0

        # T > W: choose crop
        if self.train:
            start = np.random.randint(0, T - W + 1)
        else:
            # deterministic center crop for eval
            start = max(0, (T - W) // 2)
        end = start + W
        return arr[start:end], start

    def __getitem__(self, idx: int):
        meta = self.samples[idx]
        arr = np.load(meta["path"], allow_pickle=True).astype(np.float32)  # [T, 46]
        window, start = self._crop_or_pad(arr)  # [W, 46], start index

        skeleton = torch.from_numpy(window)  # [W, 46]
        label = int(meta["label_idx"])
        seq_id = meta["seq_id"]

        if not self.with_images:
            return {"skeleton": skeleton, "label": label, "seq_id": seq_id, "start": start}

        # When with_images=True, also load aligned frames from the corresponding video.
        import cv2
        from PIL import Image as PILImage
        import torchvision.transforms as T

        if self.video_dir is None:
            raise ValueError("video_dir must be provided when with_images=True")

        video_path = os.path.join(self.video_dir, f"{seq_id}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Fall back: return zeros for images
            images = torch.zeros(self.window_size, 3, 448, 448, dtype=torch.float32)
            return {
                "skeleton": skeleton,
                "label": label,
                "images": images,
                "seq_id": seq_id,
                "start": start,
            }

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = self.window_size
        frame_indices = [min(start + k, max(0, frame_count - 1)) for k in range(W)]

        # Default transform if none provided
        if self.image_transform is None:
            self.image_transform = T.Compose(
                [
                    T.Resize((448, 448)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        frames = []
        last_valid = None
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                # Use last valid frame or zeros
                if last_valid is None:
                    frame = np.zeros((448, 448, 3), dtype=np.uint8)
                else:
                    frame = last_valid
            last_valid = frame

            # Convert BGR to RGB, then to tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(frame_rgb)
            tensor_img = self.image_transform(pil_img)  # [3, 448, 448]
            frames.append(tensor_img)

        cap.release()

        images = torch.stack(frames, dim=0)  # [W, 3, 448, 448]
        return {
            "skeleton": skeleton,
            "label": label,
            "images": images,
            "seq_id": seq_id,
            "start": start,
        }





