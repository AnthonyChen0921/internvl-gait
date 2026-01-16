import os
import json
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

# Match previous image-only setup
WINDOW_SIZE = 32
BATCH_SIZE = 1  # we will handle one video sequence at a time
EPOCHS = 20
LR = 5e-4

# To control memory usage inside the language model, we (1) limit how many
# frames of skeleton text we include, and (2) cap the total number of text
# tokens passed to the tokenizer.
MAX_SKELETON_TEXT_FRAMES = 16
MAX_TEXT_TOKENS = 1024

CKPT_PATH = "best_image_only_original_internvl_skeltext.pt"
STATE_PATH = "image_only_original_internvl_skeltext_state.pt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "GAVD-sequences")
HSMR_TEXT_DIR = os.path.join(BASE_DIR, "GAVD-HSMR-text")


def build_dataloaders() -> Tuple[DataLoader, DataLoader, List[Dict]]:
    samples = collect_labeled_sequences()
    train_samples, test_samples = video_level_train_test_split(samples, train_ratio=0.8)

    # Video-level summary
    train_videos = {s["video_id"] for s in train_samples}
    test_videos = {s["video_id"] for s in test_samples}
    print(f"\nTrain sequences: {len(train_samples)}")
    print(f"Test sequences: {len(test_samples)}")
    print(f"Train videos: {len(train_videos)}")
    print(f"Test videos: {len(test_videos)}")
    print(f"Overlap videos between train/test: {len(train_videos & test_videos)}")

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

    return train_loader, test_loader, train_samples


def compute_class_weights(train_samples: List[Dict]) -> torch.Tensor:
    counts = [0] * len(TOP7_LABELS)
    for s in train_samples:
        counts[s["label_idx"]] += 1

    counts_tensor = torch.tensor(counts, dtype=torch.float32)
    counts_tensor = torch.clamp(counts_tensor, min=1.0)
    N = counts_tensor.sum()
    K = float(len(TOP7_LABELS))
    weights = N / (K * counts_tensor)
    weights = weights / weights.mean()
    print("\nClass weights:", weights.tolist())
    return weights


def _load_skeleton_text(seq_id: str, start: int, window_size: int) -> str:
    """
    Load per-frame skeleton text from GAVD-HSMR-text/HSMR-{seq_id}.jsonl and
    align it with the same temporal window [start, start+window_size).
    """
    path = os.path.join(HSMR_TEXT_DIR, f"HSMR-{seq_id}.jsonl")
    if not os.path.exists(path):
        return ""

    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(obj)
    except OSError:
        return ""

    if not records:
        return ""

    max_idx = len(records) - 1

    # We only keep at most MAX_SKELETON_TEXT_FRAMES descriptions to avoid
    # blowing up the LLM sequence length. If there are more frames in the
    # window, we subsample them roughly uniformly.
    num_available = min(window_size, len(records) - start if start <= max_idx else 0)
    if num_available <= 0:
        num_available = min(len(records), window_size)

    num_lines = min(num_available, MAX_SKELETON_TEXT_FRAMES)
    if num_lines <= 0:
        return ""

    step = max(1, window_size // num_lines)

    lines: List[str] = []
    used = 0
    frame_idx = start
    while used < num_lines and frame_idx <= start + window_size - 1:
        idx = min(frame_idx, max_idx)
        rec = records[idx]
        skel_str = rec.get("skel", "")
        frame_no = rec.get("frame", idx)
        if skel_str:
            lines.append(f"Frame {frame_no}: {skel_str}")
            used += 1
        frame_idx += step

    return "\n".join(lines)


def build_prompt(skeleton_text: str) -> str:
    base_prompt = (
        "You are an expert gait clinician. Based on the available gait information, "
        "classify the patient's gait pattern.\n\n"
        "Gait pattern definitions:\n"
        "- abnormal: any gait pattern that deviates from normal but does not fit the specific patterns below.\n"
        "- myopathic: waddling or Trendelenburg-type gait due to proximal muscle weakness.\n"
        "- exercise: exaggerated, energetic, or performance-like gait related to sport or exercise.\n"
        "- normal: typical, symmetric gait without obvious abnormalities.\n"
        "- style: exaggerated or stylistic walking pattern without clear neurological or orthopedic cause.\n"
        "- cerebral palsy: spastic, scissoring, toe-walking, or crouched gait typical of cerebral palsy.\n"
        "- parkinsons: shuffling, stooped posture, reduced arm swing, and festination typical of Parkinson's disease.\n\n"
        "Below are per-frame skeleton parameters extracted from the same gait sequence.\n"
        "Use both the visual gait information and these skeleton parameters to internally decide which class is most likely. "
        "You do not need to output the class name."
    )

    if skeleton_text:
        return base_prompt + "\n\nSkeleton parameters:\n" + skeleton_text

    return base_prompt + "\n\n(No skeleton parameters available for this sequence.)"


def prepare_batch(batch: Dict, tokenizer, device: str):
    """
    Convert a batch from GavdSkeletonDataset into pixel_values + text inputs.

    batch['images']: [B, W, 3, H, W]
    batch['label']:  [B]
    """
    images = batch["images"]  # [B, W, 3, H, W]
    labels = batch["label"].to(device)
    seq_id = batch["seq_id"][0]
    start = int(batch["start"][0].item())

    # We currently use BATCH_SIZE = 1, so treat each sequence as one multimodal sample.
    # Collapse the batch dimension and keep all frames as a sequence of images.
    # Result: [W, 3, H, W], as used in the zero-shot video script.
    pixel_values = images.squeeze(0).to(device)

    skel_text = _load_skeleton_text(seq_id, start, WINDOW_SIZE)
    prompt = build_prompt(skel_text)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    return pixel_values, labels, input_ids, attention_mask


@torch.no_grad()
def extract_features(model, tokenizer, batch: Dict, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run InternVL using (almost) its original multimodal path:
      - Build a prompt containing a single <image> placeholder.
      - Expand that into IMG_START / IMG_CONTEXT / IMG_END tokens, with
        num_image_token * num_frames context tokens.
      - Encode images via model.extract_feature and inject those visual tokens
        at IMG_CONTEXT positions in the language model input embeddings.
      - Run the frozen language model and take the last token as the video feature.
    """
    # Unpack batch
    images = batch["images"]  # [B, W, 3, H, W]
    labels = batch["label"].to(device)

    # BATCH_SIZE = 1 → treat each sequence as one multimodal sample with W frames.
    pixel_values = images.squeeze(0)

    # Match model dtype / device
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    pixel_values = pixel_values.to(device=device, dtype=model_dtype)  # [W, 3, H, W]

    # Build question with an explicit <image> placeholder, similar to chat()
    prompt = build_prompt()
    question = "<image>\n" + prompt

    # InternVL special tokens and config
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    # Number of images (frames) in this sequence
    num_patches = pixel_values.shape[0]

    # num_image_token is defined on the InternVL chat model
    num_image_token = getattr(model, "num_image_token", 1)

    # Construct the expanded image token sequence that replaces <image>
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (num_image_token * num_patches) + IMG_END_TOKEN
    query = question.replace("<image>", image_tokens, 1)

    # Tokenize the final query
    model_inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
    )
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    # Set img_context_token_id as in chat()
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # Encode images with the official vision path
    vit_embeds = model.extract_feature(pixel_values)  # [total_imgs, T_v, D]

    # Build LM input embeddings and inject visual tokens at IMG_CONTEXT positions
    language_model = getattr(model, "language_model", model)
    input_embeds = language_model.get_input_embeddings()(input_ids)  # [1, N, D]

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    flat_ids = input_ids.reshape(B * N)
    selected = flat_ids == img_context_token_id

    # Reshape visual embeddings to match the number of IMG_CONTEXT positions
    vit_flat = vit_embeds.reshape(-1, C).to(input_embeds.device)
    if vit_flat.size(0) >= selected.sum():
        input_embeds[selected] = vit_flat[: selected.sum()]
    else:
        # If there are more IMG_CONTEXT positions than visual tokens (shouldn't
        # normally happen), fill as many as possible.
        input_embeds[selected.nonzero(as_tuple=True)[0][: vit_flat.size(0)]] = vit_flat

    input_embeds = input_embeds.reshape(B, N, C)

    # Run the frozen language model and take the last token as the pooled feature
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
    tokenizer, model, _ = load_model(device=DEVICE)
    # `load_model` already sets model.eval() and requires_grad_(False).
    model.to(DEVICE)

    hidden_size = getattr(model, "hidden_size", None)
    if hidden_size is None:
        # Fall back to underlying language model config
        lm = getattr(model, "language_model", model)
        hidden_size = lm.config.hidden_size

    num_classes = len(TOP7_LABELS)
    classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32).to(DEVICE)

    train_loader, test_loader, train_samples = build_dataloaders()
    class_weights = compute_class_weights(train_samples).to(DEVICE)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_macro_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.eval()  # frozen feature extractor
        classifier.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - train", leave=False):
            optimizer.zero_grad()

            with torch.no_grad():
                feats, labels = extract_features(model, tokenizer, batch, DEVICE)

            logits = classifier(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        print(f"\nEpoch {epoch}/{EPOCHS} - train loss: {train_loss:.4f}, acc: {train_acc*100:.2f}%")

        _, macro_f1 = evaluate(model, classifier, test_loader, tokenizer, DEVICE)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "macro_f1": best_macro_f1,
                    "epoch": epoch,
                },
                CKPT_PATH,
            )
            print(f"Saved new best original-InternVL (skeltext) model (macro-F1={macro_f1*100:.2f}%)")

        # Always save the latest epoch checkpoint so we can evaluate the final model.
        torch.save(
            {
                "model": model.state_dict(),
                "classifier": classifier.state_dict(),
                "macro_f1": macro_f1,
                "epoch": epoch,
            },
            STATE_PATH,
        )
        print(f"Saved last-epoch original-InternVL (skeltext) model to {STATE_PATH}")


if __name__ == "__main__":
    main()

