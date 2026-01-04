import os
import glob
import argparse

import torch
import torchvision.transforms as T
from PIL import Image

from minimal_internvl_inference import load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREVIEW_DIR = os.path.join(BASE_DIR, "GAVD-frames-preview")

TOP7_LABELS = [
    "abnormal",
    "myopathic",
    "exercise",
    "normal",
    "style",
    "cerebral palsy",
    "parkinsons",
]


def pick_preview_image(seq_id: str | None = None) -> str:
    """
    Pick one preview image from GAVD-frames-preview.

    If seq_id is provided, choose the first image whose filename starts with that seq_id.
    Otherwise, just take the first available image.
    """
    pattern = "*.jpg" if seq_id is None else f"{seq_id}_f*.jpg"
    paths = sorted(glob.glob(os.path.join(PREVIEW_DIR, pattern)))
    if not paths:
        raise FileNotFoundError(f"No preview images found in {PREVIEW_DIR} with pattern {pattern}")
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description="Zero-shot InternVL on a single gait frame")
    parser.add_argument(
        "--seq-id",
        type=str,
        default=None,
        help="Optional sequence id to pick a specific preview frame (prefix of filename in GAVD-frames-preview).",
    )
    args = parser.parse_args()

    # 1) Load InternVL model (1B or 8B depending on INTERNVL_MODEL_PATH)
    tokenizer, model, device = load_model()

    # 2) Pick a preview frame
    img_path = pick_preview_image(args.seq_id)
    print(f"Using preview image: {img_path}")

    image = Image.open(img_path).convert("RGB")

    # 3) Preprocess image to pixel_values tensor as in minimal_internvl_inference
    transform = T.Compose(
        [
            T.Resize((448, 448)),
            T.ToTensor(),  # [C, H, W] in [0, 1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, 448, 448]

    # Match the model's dtype
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
    pixel_values = pixel_values.to(device=device, dtype=model_dtype)

    # 4) Ask a zero-shot gait question with explicit label definitions
    question = (
        "You are an expert gait clinician. This frame is from a gait examination video.\n\n"
        "Gait pattern definitions:\n"
        "- abnormal: any gait pattern that deviates from normal but does not fit the specific patterns below.\n"
        "- myopathic: waddling or Trendelenburg-type gait due to proximal muscle weakness.\n"
        "- exercise: exaggerated, energetic, or performance-like gait related to sport or exercise.\n"
        "- normal: typical, symmetric gait without obvious abnormalities.\n"
        "- style: exaggerated or stylistic walking pattern without clear neurological or orthopedic cause.\n"
        "- cerebral palsy: spastic, scissoring, toe-walking, or crouched gait typical of cerebral palsy.\n"
        "- parkinsons: shuffling, stooped posture, reduced arm swing, and festination typical of Parkinson's disease.\n\n"
        "Based solely on this frame, what is the most likely gait pattern?\n"
        "Just answer with ONE of the following class names exactly:\n"
        "abnormal, myopathic, exercise, normal, style, cerebral palsy, parkinsons."
    )
    print(f"\nQuestion: {question}\n")

    # Most InternVL3.5 chat implementations accept a PIL image via `pixel_values` or `images` kwarg.
    # We can reuse the same `chat` interface used in minimal_internvl_inference.run_image_text_demo.
    generation_config = {
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 0.7,
    }

    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
        )

    # Post-process: extract the first matching class label, if any
    resp_lower = response.lower()
    predicted = None
    for label in TOP7_LABELS:
        if label in resp_lower:
            predicted = label
            break

    print("Model output (raw):\n")
    print(response)
    print("\nPredicted gait pattern:")
    print(predicted if predicted is not None else response)


if __name__ == "__main__":
    main()


