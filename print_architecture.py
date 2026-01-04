# save as: print_architecture.py
import os
import torch

from minimal_internvl_inference import load_model
from internvl_skeleton_adapter import InternVLWithSkeleton

def count_parameters(module, trainable_only=False):
    params = [
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    ]
    return sum(params)

def main():
    # Point to 1B model by env var or hardcode the path
    os.environ["INTERNVL_MODEL_PATH"] = r"C:\Users\1nkas-Strix-4090-ll\Models\InternVL3_5-1B"
    os.environ["HF_HUB_OFFLINE"] = "1"

    tokenizer, base_model, device = load_model()
    print(f"Loaded base InternVL model on device: {device}")
    print()

    skel_model = InternVLWithSkeleton(base_model).to(device)
    skel_model.eval()

    print("=== InternVLWithSkeleton architecture ===")
    print(skel_model)
    print()

    # High-level parameter counts
    total_params = count_parameters(skel_model, trainable_only=False)
    trainable_params = count_parameters(skel_model, trainable_only=True)
    frozen_params = total_params - trainable_params

    print(f"Total parameters (adapter + backbone): {total_params:,}")
    print(f"Trainable parameters (adapter + classifier only): {trainable_params:,}")
    print(f"Frozen parameters (InternVL backbone): {frozen_params:,}")
    print()

    print("=== Key submodules ===")
    print("Language model:", skel_model.language_model.__class__.__name__)
    print("Skeleton projection:", skel_model.skel_proj)
    print("Skeleton positional embeddings:", skel_model.skel_pos_emb.shape)
    print("Skeleton LayerNorm:", skel_model.skel_ln)
    print()

    # Dummy forward to show token dimensions
    B, T = 2, 64
    dummy_skel = torch.randn(B, T, skel_model.skel_dim, device=device)
    dummy_ids = torch.randint(0, tokenizer.vocab_size, (B, 16), device=device)
    dummy_mask = torch.ones_like(dummy_ids, device=device)

    with torch.no_grad():
        out = skel_model(
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            skeleton_feats=dummy_skel,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]  # [B, T_prefix+L, D]
        print("Last hidden state shape (skeleton+text):", hidden.shape)

if __name__ == "__main__":
    main()