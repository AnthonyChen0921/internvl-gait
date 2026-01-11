import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from minimal_internvl_inference import load_model
from internvl_temporal_adapter import EVLTemporalDecoder
from gavd_skeleton_dataset import (
    GavdSkeletonDataset,
    collect_labeled_sequences,
    video_level_train_test_split,
    TOP7_LABELS,
)
from train_evl_compact_classifier import (
    WINDOW_SIZE,
    BATCH_SIZE,
    VIDEO_DIR,
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
    if not os.path.exists(STATE_PATH):
        raise FileNotFoundError(
            f"{STATE_PATH} not found. Run train_evl_compact_classifier.py to create it "
            "and make sure training finishes."
        )

    # Load backbone and EVL decoder definition
    tokenizer, base_model, _ = load_model(device=DEVICE)  # tokenizer unused

    decoder = EVLTemporalDecoder(
        base_model,
        max_frames=WINDOW_SIZE,
        num_queries=8,
        num_layers=3,
        num_heads=4,
    ).to(DEVICE)

    hidden_size = decoder.hidden_size
    num_classes = len(TOP7_LABELS)
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    state = torch.load(STATE_PATH, map_location=DEVICE)
    decoder.load_state_dict(state["decoder"])
    classifier.load_state_dict(state["classifier"])

    print(
        f"Loaded training state from {STATE_PATH}: "
        f"epoch={state.get('epoch')}, best_macro_f1={state.get('best_macro_f1', 0.0)*100:.2f}%"
    )

    test_loader = build_test_loader()
    evaluate(decoder, classifier, test_loader, DEVICE)


if __name__ == "__main__":
    main()





