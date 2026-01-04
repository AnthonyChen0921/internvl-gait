import torch

from minimal_internvl_inference import load_model
from internvl_temporal_adapter import TemporalVideoEncoder
from gavd_skeleton_dataset import (
    GavdSkeletonDataset,
    collect_labeled_sequences,
    video_level_train_test_split,
)
from train_skeleton_image_classifier import TOP7_LABELS, evaluate, WINDOW_SIZE, BATCH_SIZE, VIDEO_DIR


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "best_skeleton_image_classifier.pt"


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
    # Load backbone + tokenizer
    tokenizer, base_model, _ = load_model(device=DEVICE)
    video_encoder = TemporalVideoEncoder(
        base_model,
        max_frames=WINDOW_SIZE,
        use_skeleton=True,
    ).to(DEVICE)
    video_encoder.eval()

    # Classifier head (same size as in training)
    hidden_size = video_encoder.hidden_size
    num_classes = len(TOP7_LABELS)
    classifier = torch.nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    video_encoder.load_state_dict(ckpt["video_encoder"])
    classifier.load_state_dict(ckpt["classifier"])
    print(
        f"Loaded checkpoint from {CKPT_PATH} "
        f"(epoch {ckpt.get('epoch')}, macro-F1={ckpt.get('macro_f1', 0.0) * 100:.2f}%)"
    )

    test_loader = build_test_loader()
    evaluate(video_encoder, classifier, test_loader, DEVICE)


if __name__ == "__main__":
    main()





