import os
import argparse

import torch

from minimal_internvl_inference import load_model
from internvl_skeleton_adapter import InternVLWithSkeleton
from gavd_skeleton_dataset import (
    GavdSkeletonDataset,
    collect_labeled_sequences,
    video_level_train_test_split,
)
from train_image_only_classifier import (
    TOP7_LABELS,
    evaluate,
    WINDOW_SIZE,
    BATCH_SIZE,
    VIDEO_DIR,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_image_only_classifier.pt"
STATE_PATH = "image_only_train_state.pt"


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
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-last",
        action="store_true",
        help="Use the last-epoch checkpoint (image_only_train_state.pt) instead of the best checkpoint.",
    )
    args = parser.parse_args()

    # Load backbone + tokenizer
    tokenizer, base_model, _ = load_model(device=DEVICE)
    img_model = InternVLWithSkeleton(base_model).to(DEVICE)
    img_model.eval()

    # Classifier head (same size as in training)
    hidden_size = img_model.hidden_size
    num_classes = len(TOP7_LABELS)
    classifier = torch.nn.Linear(
        hidden_size,
        num_classes,
        dtype=torch.float32,
    ).to(DEVICE)

    # Choose which checkpoint to load
    ckpt_path = CKPT_PATH
    if args.use_last:
        if not os.path.exists(STATE_PATH):
            raise FileNotFoundError(
                f"--use-last was specified, but {STATE_PATH} does not exist. "
                f"Run train_image_only_classifier.py first to create it."
            )
        ckpt_path = STATE_PATH

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    img_model.load_state_dict(ckpt["image_model"])
    classifier.load_state_dict(ckpt["classifier"])
    print(
        f"Loaded checkpoint from {CKPT_PATH} "
        f"(epoch {ckpt.get('epoch')}, macro-F1={ckpt.get('macro_f1', 0.0) * 100:.2f}%)"
    )

    test_loader = build_test_loader()
    evaluate(img_model, classifier, test_loader, tokenizer, DEVICE)


if __name__ == "__main__":
    main()

















