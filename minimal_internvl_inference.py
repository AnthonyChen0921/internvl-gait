import os
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image


MODEL_NAME = os.environ.get("INTERNVL_MODEL_PATH", "OpenGVLab/InternVL3_5-8B")


def load_model(device: Optional[str] = None):
    """
    Load InternVL in inference mode with remote code enabled.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id_or_path: Path | str
    # If MODEL_NAME points to an existing local directory, use a Path object to
    # force Transformers to treat it as a local path (and not as a repo id).
    if os.path.isdir(MODEL_NAME):
        model_id_or_path = Path(MODEL_NAME)
    else:
        model_id_or_path = MODEL_NAME

    print(f"Loading model {model_id_or_path} on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)

    # Choose dtype / device mapping based on model size.
    model_id_str = str(model_id_or_path)
    if "InternVL3_5-8B" in model_id_str and device == "cuda":
        # For 8B, prefer bfloat16 with automatic device placement to fit in GPU memory.
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        # For smaller models (e.g., 1B), we can keep everything in float32 on a 24GB GPU.
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(device)

    model.eval()
    model.requires_grad_(False)

    return tokenizer, model, device


def load_demo_image() -> Image.Image:
    """
    Create a simple local demo image (no network required).
    """
    img = Image.new("RGB", (256, 256), color=(200, 200, 200))
    return img


def run_text_only_demo(tokenizer, model, device: str):
    prompt = "Describe what InternVL is capable of in one short sentence."
    print(f"\n=== Text-only demo ===\nPrompt: {prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # InternVL overrides `generate` to expect image context tokens even for
    # text-only use. For a minimal sanity check, call the underlying language
    # model's `generate` directly if available.
    gen_model = getattr(model, "language_model", model)

    with torch.no_grad():
        output = gen_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Output:\n", text)


def run_image_text_demo(tokenizer, model, device: str):
    """
    Minimal image+text demo using InternVL's custom `chat` interface.
    """
    print("\n=== Image+text demo ===")
    image = load_demo_image()
    question = "What is in this picture?"
    print(f"Question: {question}")

    # Preprocess image according to preprocessor_config (448x448, normalize, channels_first).
    transform = T.Compose(
        [
            T.Resize((448, 448)),
            T.ToTensor(),  # [C, H, W] in [0, 1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, 448, 448]

    # Match the model's dtype (InternVL is usually in float16 on CUDA).
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
    pixel_values = pixel_values.to(device=device, dtype=model_dtype)

    # Use InternVL's custom `chat` method, which correctly inserts image context tokens
    # and calls its overridden generate().
    generation_config = {
        "max_new_tokens": 64,
        "do_sample": False,
    }

    with torch.no_grad():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
        )

    print("Output:\n", response)


def main():
    tokenizer, model, device = load_model()

    # 1) Verify plain text inference
    run_text_only_demo(tokenizer, model, device)

    # 2) Try image+text inference
    try:
        run_image_text_demo(tokenizer, model, device)
    except Exception as e:
        # If the image API differs, we still have the text-only path working.
        print("\nImage+text demo failed with error (this is OK as long as text-only works first):")
        print(repr(e))


if __name__ == "__main__":
    main()


