import os
import glob
import argparse

import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HSMR_SINGLE_DIR = os.path.join(BASE_DIR, "GAVD-HSMR-single")
VIDEO_DIR = os.path.join(BASE_DIR, "GAVD-sequences")

DEFAULT_FRAMES = 16  # use fewer frames to stay within InternVL's max context


def pick_sample_seq(index: int = 0) -> str:
    """
    Pick a sequence id from the available HSMR-single files.

    index: which file to pick (0-based). This makes it easy to \"resume\" by
    passing a different index if you quit and want to test another sequence.
    """
    paths = sorted(glob.glob(os.path.join(HSMR_SINGLE_DIR, "HSMR-*.npy")))
    if not paths:
        raise FileNotFoundError(f"No npy files found in {HSMR_SINGLE_DIR}")
    if index < 0 or index >= len(paths):
        raise IndexError(f"Index {index} is out of range for {len(paths)} sequences")
    fname = os.path.basename(paths[index])
    seq_id = fname[len("HSMR-") : -len(".npy")]
    print(f"Total sequences available: {len(paths)}; using index {index}")
    return seq_id


def load_video_window(seq_id: str, device: str, dtype: torch.dtype, window_size: int = DEFAULT_FRAMES) -> torch.Tensor:
    video_path = os.path.join(VIDEO_DIR, f"{seq_id}.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = window_size

    # Sample W frames evenly across the video
    if frame_count <= 0:
        raise RuntimeError(f"No frames found in {video_path}")

    if frame_count <= W:
        indices = list(range(frame_count)) + [frame_count - 1] * (W - frame_count)
    else:
        step = frame_count / W
        indices = [int(i * step) for i in range(W)]

    transform = T.Compose(
        [
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frames = []
    indices_iter = tqdm(indices, desc="Sampling video frames", leave=False)
    last_valid = None
    for fi in indices_iter:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            if last_valid is None:
                frame = np.zeros((448, 448, 3), dtype=np.uint8)
            else:
                frame = last_valid
        last_valid = frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        tensor = transform(pil)  # [3, 448, 448]
        frames.append(tensor)

    cap.release()

    pixel_values = torch.stack(frames, dim=0).to(device=device, dtype=dtype)  # [W, 3, 448, 448]
    return pixel_values


def main():
    parser = argparse.ArgumentParser(description="Zero-shot InternVL video-only gait classification")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Which sequence index to use from GAVD-HSMR-single (0-based). "
             "Use a different index to 'resume' testing on another sequence.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"Number of frames to sample from the video (default: {DEFAULT_FRAMES}). "
             "Using too many frames may exceed InternVL's context length.",
    )
    args = parser.parse_args()

    tokenizer, model, device = load_model()

    print("Zero-shot video experiment: image sequence + text prompt (no skeleton).")

    seq_id = pick_sample_seq(args.index)
    print(f"Using sequence id: {seq_id}")

    # Determine model dtype
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32

    pixel_values = load_video_window(seq_id, device, model_dtype, window_size=args.frames)  # [W, 3, 448, 448]

    # Build expert prompt with label definitions
    prompt = (
        "You are an expert gait clinician. Based on the sequence of gait images, "
        "classify the patient's gait pattern.\n\n"
        "Gait pattern definitions:\n"
        "- abnormal: any gait pattern that deviates from normal but does not fit the specific patterns below.\n"
        "- myopathic: waddling or Trendelenburg-type gait due to proximal muscle weakness.\n"
        "- exercise: exaggerated, energetic, or performance-like gait related to sport or exercise.\n"
        "- normal: typical, symmetric gait without obvious abnormalities.\n"
        "- style: exaggerated or stylistic walking pattern without clear neurological or orthopedic cause.\n"
        "- cerebral palsy: spastic, scissoring, toe-walking, or crouched gait typical of cerebral palsy.\n"
        "- parkinsons: shuffling, stooped posture, reduced arm swing, and festination typical of Parkinson's disease.\n\n"
        "Based solely on this image sequence, what is the most likely gait pattern?\n"
        "Just answer with ONE of the following class names exactly:\n"
        "abnormal, myopathic, exercise, normal, style, cerebral palsy, parkinsons."
    )

    generation_config = {
        "max_new_tokens": 64,
        "do_sample": False,
    }

    print("Generating zero-shot prediction...")
    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=generation_config,
        )

    print("\nModel output:\n")
    print(response)


if __name__ == "__main__":
    main()


