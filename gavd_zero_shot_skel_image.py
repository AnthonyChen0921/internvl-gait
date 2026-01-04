import os
import glob

import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

from minimal_internvl_inference import load_model
from internvl_skeleton_adapter import InternVLWithSkeleton


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HSMR_SINGLE_DIR = os.path.join(BASE_DIR, "GAVD-HSMR-single")
VIDEO_DIR = os.path.join(BASE_DIR, "GAVD-sequences")

TOP7_LABELS = [
    "abnormal",
    "myopathic",
    "exercise",
    "normal",
    "style",
    "cerebral palsy",
    "parkinsons",
]

WINDOW_SIZE = 64


def pick_sample_seq() -> str:
    paths = sorted(glob.glob(os.path.join(HSMR_SINGLE_DIR, "HSMR-*.npy")))
    if not paths:
        raise FileNotFoundError(f"No npy files found in {HSMR_SINGLE_DIR}")
    fname = os.path.basename(paths[0])
    seq_id = fname[len("HSMR-") : -len(".npy")]
    return seq_id


def load_skeleton_window(seq_id: str, device: str) -> torch.Tensor:
    path = os.path.join(HSMR_SINGLE_DIR, f"HSMR-{seq_id}.npy")
    arr = np.load(path, allow_pickle=True).astype(np.float32)  # [T, 46]
    T_total = arr.shape[0]
    W = WINDOW_SIZE
    if T_total <= W:
        pad = np.zeros((W - T_total, arr.shape[1]), dtype=arr.dtype)
        window = np.concatenate([arr, pad], axis=0)
    else:
        start = max(0, (T_total - W) // 2)
        end = start + W
        window = arr[start:end]
    skel = torch.from_numpy(window).unsqueeze(0).to(device)  # [1, W, 46]
    return skel


def load_image_window(seq_id: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    video_path = os.path.join(VIDEO_DIR, f"{seq_id}.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = WINDOW_SIZE

    # Sample W frames evenly across the video
    indices = []
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
    last_valid = None
    for fi in indices:
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

    images = torch.stack(frames, dim=0).unsqueeze(0).to(device=device, dtype=dtype)  # [1, W, 3, 448, 448]
    return images


def main():
    tokenizer, base_model, device = load_model()
    model = InternVLWithSkeleton(base_model).to(device)
    model.eval()

    # Pick a sample sequence
    seq_id = pick_sample_seq()
    print(f"Using sequence: {seq_id}")

    # Prepare skeleton and images
    skeleton_feats = load_skeleton_window(seq_id, device)  # [1, W, 46]
    lm_dtype = next(model.language_model.parameters()).dtype
    pixel_values = load_image_window(seq_id, device, lm_dtype)  # [1, W, 3, 448, 448]

    # Build prompt with label definitions, ask for one class name
    prompt = (
        "You are an expert gait clinician. Based on the sequence of gait images and 3D skeleton parameters, "
        "classify the patient's gait pattern.\n\n"
        "Gait pattern definitions:\n"
        "- abnormal: any gait pattern that deviates from normal but does not fit the specific patterns below.\n"
        "- myopathic: waddling or Trendelenburg-type gait due to proximal muscle weakness.\n"
        "- exercise: exaggerated, energetic, or performance-like gait related to sport or exercise.\n"
        "- normal: typical, symmetric gait without obvious abnormalities.\n"
        "- style: exaggerated or stylistic walking pattern without clear neurological or orthopedic cause.\n"
        "- cerebral palsy: spastic, scissoring, toe-walking, or crouched gait typical of cerebral palsy.\n"
        "- parkinsons: shuffling, stooped posture, reduced arm swing, and festination typical of Parkinson's disease.\n\n"
        "Based on this multimodal information, what is the most likely gait pattern?\n"
        "Just answer with ONE of the following class names exactly:\n"
        "abnormal, myopathic, exercise, normal, style, cerebral palsy, parkinsons."
    )

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Manually construct inputs_embeds with image + skeleton + text tokens
    with torch.no_grad():
        lm = model.language_model
        text_embeds = lm.get_input_embeddings()(input_ids)        # [1, L, D]
        img_tokens = model.encode_images(pixel_values)            # [1, W, D]
        skel_tokens = model.encode_skeleton(skeleton_feats)       # [1, W, D]

        prefix = torch.cat([img_tokens, skel_tokens], dim=1)      # [1, 2W, D]
        prefix_mask = torch.ones(1, prefix.size(1), dtype=attention_mask.dtype, device=device)

        inputs_embeds = torch.cat([prefix, text_embeds], dim=1)   # [1, 2W + L, D]
        fused_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        output_ids = lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=fused_mask,
            max_new_tokens=8,
            do_sample=False,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    text_lower = text.lower()
    predicted = None
    for label in TOP7_LABELS:
        if label in text_lower:
            predicted = label
            break

    print("\nRaw generation:\n", text)
    print("\nPredicted gait pattern (from skeleton + image + text):")
    print(predicted if predicted is not None else text)


if __name__ == "__main__":
    main()

























