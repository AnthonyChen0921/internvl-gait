import os
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from minimal_internvl_inference import load_model
from gavd_skeleton_dataset import (
    GavdSkeletonDataset,
    collect_labeled_sequences,
    video_level_train_test_split,
    TOP7_LABELS,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 32
BATCH_SIZE = 1

CKPT_PATH = "best_image_only_original_internvl_instruction.pt"
VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GAVD-sequences")


def build_dataloaders() -> Tuple[DataLoader, DataLoader]:
    samples = collect_labeled_sequences()
    train_samples, test_samples = video_level_train_test_split(samples, train_ratio=0.8)

    train_ds = GavdSkeletonDataset(
        train_samples,
        window_size=WINDOW_SIZE,
        train=True,
        with_images=True,
        video_dir=VIDEO_DIR,
    )
    test_ds = GavdSkeletonDataset(
        test_samples,
        window_size=WINDOW_SIZE,
        train=False,
        with_images=True,
        video_dir=VIDEO_DIR,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def build_prompt() -> str:
    return (
        "You are an expert gait clinician. Based on the available gait information, "
        "classify the patient's gait pattern."
    )


@torch.no_grad()
def extract_features(model, tokenizer, batch: Dict, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    images = batch["images"]  # [B, W, 3, H, W]
    labels = batch["label"].to(device)

    pixel_values = images.squeeze(0)

    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    pixel_values = pixel_values.to(device=device, dtype=model_dtype)

    prompt = build_prompt()
    question = "<image>\n" + prompt

    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    num_patches = pixel_values.shape[0]
    num_image_token = getattr(model, "num_image_token", 1)

    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (num_image_token * num_patches) + IMG_END_TOKEN
    query = question.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    vit_embeds = model.extract_feature(pixel_values)

    language_model = getattr(model, "language_model", model)
    input_embeds = language_model.get_input_embeddings()(input_ids)

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    flat_ids = input_ids.reshape(B * N)
    selected = flat_ids == img_context_token_id

    vit_flat = vit_embeds.reshape(-1, C).to(input_embeds.device)
    if vit_flat.size(0) >= selected.sum():
        input_embeds[selected] = vit_flat[: selected.sum()]
    else:
        input_embeds[selected.nonzero(as_tuple=True)[0][: vit_flat.size(0)]] = vit_flat

    input_embeds = input_embeds.reshape(B, N, C)

    outputs = language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden = outputs.hidden_states[-1]  # [1, N, D]
    feats = hidden[:, -1, :].float()  # [1, D]

    return feats, labels


def evaluate(model, classifier, data_loader, tokenizer, device: str):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0

    num_classes = len(TOP7_LABELS)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            feats, labels = extract_features(model, tokenizer, batch, device)
            logits = classifier(feats)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for c in range(num_classes):
                tp[c] += ((preds == c) & (labels == c)).sum().item()
                fp[c] += ((preds == c) & (labels != c)).sum().item()
                fn[c] += ((preds != c) & (labels == c)).sum().item()

    acc = correct / max(1, total)

    per_class_f1 = []
    for c in range(num_classes):
        precision = tp[c] / max(1, tp[c] + fp[c])
        recall = tp[c] / max(1, tp[c] + fn[c])
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class_f1.append(f1)

    macro_f1 = sum(per_class_f1) / num_classes

    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"Test macro-F1: {macro_f1 * 100:.2f}%")
    print("Per-class F1:")
    for idx, f1 in enumerate(per_class_f1):
        print(f"  {TOP7_LABELS[idx]}: {f1 * 100:.2f}%")

    return acc, macro_f1


def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"{CKPT_PATH} not found. Run training first.")

    tokenizer, model, _ = load_model(device=DEVICE)
    model.to(DEVICE)

    hidden_size = getattr(model, "hidden_size", None)
    if hidden_size is None:
        lm = getattr(model, "language_model", model)
        hidden_size = lm.config.hidden_size

    num_classes = len(TOP7_LABELS)
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    classifier.load_state_dict(ckpt["classifier"])
    print(
        f"Loaded checkpoint from {CKPT_PATH} "
        f"(epoch {ckpt.get('epoch')}, macro-F1={ckpt.get('macro_f1', 0.0) * 100:.2f}%)"
    )

    _, test_loader = build_dataloaders()
    evaluate(model, classifier, test_loader, tokenizer, DEVICE)


if __name__ == "__main__":
    main()

