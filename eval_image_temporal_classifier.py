import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from internvl_temporal_adapter import TemporalVideoEncoder
from gavd_skeleton_dataset import (
    GavdSkeletonDataset,
    collect_labeled_sequences,
    video_level_train_test_split,
    TOP7_LABELS,
)
from train_image_temporal_classifier import (
    WINDOW_SIZE,
    BATCH_SIZE,
    VIDEO_DIR,
    CKPT_PATH,
    STATE_PATH,
    collate_fn,
    evaluate,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_test_loader():
    samples = collect_labeled_sequences()
    _, test_samples = video_level_train_test_split(samples, train_ratio=0.8)

    test_ds = GavdSkeletonDataset(
        test_samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=VIDEO_DIR,
    )

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return test_loader


def main():
    # Prefer evaluating the final training state (e.g., epoch 20).
    if not os.path.exists(STATE_PATH):
        raise FileNotFoundError(
            f"{STATE_PATH} not found. Run train_image_temporal_classifier.py to create it "
            "and make sure training reaches epoch 20."
        )

    # Load backbone and temporal encoder definition
    tokenizer, base_model, _ = load_model(device=DEVICE)  # tokenizer unused

    video_encoder = TemporalVideoEncoder(
        base_model,
        max_frames=WINDOW_SIZE,
        use_skeleton=False,
    ).to(DEVICE)

    hidden_size = video_encoder.hidden_size
    num_classes = len(TOP7_LABELS)
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    state = torch.load(STATE_PATH, map_location=DEVICE)
    video_encoder.load_state_dict(state["video_encoder"])
    classifier.load_state_dict(state["classifier"])

    print(
        f"Loaded training state from {STATE_PATH}: "
        f"epoch={state.get('epoch')}, best_macro_f1={state.get('best_macro_f1', 0.0)*100:.2f}%"
    )

    test_loader = build_test_loader()
    evaluate(video_encoder, classifier, test_loader, DEVICE)


if __name__ == "__main__":
    main()






