import torch


def mean_pooling_flow_demo(
    batch_size: int = 2,
    num_frames: int = 32,
    num_patches: int = 576,
    hidden_dim: int = 1024,
    text_len: int = 200,
):
    """
    Demonstrate (with shapes only) how the mean-pooling image experiment processes
    images from patches -> frame tokens -> LLM -> final mean over frames.
    This mirrors the flow used in `train_image_only_classifier.py`.
    """
    B = batch_size
    T = num_frames
    N = num_patches
    D = hidden_dim
    L = text_len

    print("Legend: ❄️ frozen module (not trainable), 🔧 trainable module\n")

    print("=== Step 1: Raw video batch ===")
    images = torch.randn(B, T, 3, 224, 224)
    print(f"images:           {tuple(images.shape)}  [B, T, C, H, W]")

    print("\n=== Step 2: Flatten time for vision encoder ===")
    images_bt = images.view(B * T, 3, 224, 224)
    print(f"images_bt:        {tuple(images_bt.shape)}  [B*T, C, H, W]")

    print("\n=== Step 3: Vision encoder outputs patch tokens ===")
    # In the real experiment, InternVL's vision encoder (InternViT) produces
    # one token per image patch. Here we simulate that with random tensors.
    vit_tokens = torch.randn(B * T, N, D)
    print(f"vit_tokens:       {tuple(vit_tokens.shape)}  [B*T, N_patch, D]")

    print("\n=== Step 4: Mean over patches -> one token per frame ===")
    frame_tokens_bt = vit_tokens.mean(dim=1)  # collapse patches
    print(f"frame_tokens_bt:  {tuple(frame_tokens_bt.shape)}  [B*T, D]")
    frame_tokens = frame_tokens_bt.view(B, T, D)
    print(f"frame_tokens:     {tuple(frame_tokens.shape)}  [B, T, D] (one token per frame)")

    print("\n=== Step 5: Concatenate frame tokens with text embeddings ===")
    text_tokens = torch.randn(B, L, D)
    print(f"text_tokens:      {tuple(text_tokens.shape)}  [B, L_text, D]")
    llm_input = torch.cat([frame_tokens, text_tokens], dim=1)
    print(f"llm_input:        {tuple(llm_input.shape)}  [B, T + L_text, D]")

    print("\n=== Step 6: LLM processes the sequence ===")
    # In the real code, this is the last hidden state from the language model.
    # Here we simply pretend the LLM is identity to keep the demo lightweight.
    hidden = llm_input
    print(f"hidden:           {tuple(hidden.shape)}  [B, T + L_text, D]")

    print("\n=== Step 7: Take the image part and mean over frames ===")
    img_hidden = hidden[:, :T, :]
    print(f"img_hidden:       {tuple(img_hidden.shape)}  [B, T, D]")
    pooled_feats = img_hidden.mean(dim=1)
    print(f"pooled_feats:     {tuple(pooled_feats.shape)}  [B, D] (mean over frames)")

    print("\n=== Summary (collapses) ===")
    print("1) Vision encoder ❄️: [B, T, C, H, W] -> [B*T, N_patch, D]  (frozen InternViT)")
    print("2) Spatial mean:       [B*T, N_patch, D] --mean over patches--> [B, T, D]")
    print("3) LLM output ❄️:     [B, T + L_text, D] -> take first T tokens -> [B, T, D]  (frozen Qwen3)")
    print("4) Temporal mean 🔧:  [B, T, D] --mean over frames--> [B, D]  (feature fed to trainable classifier)")

    print("\nThis matches the mean-pooling classifier path: "
          "collapse patches first, then collapse frames.")


def print_ascii_diagram():
    """
    Optional: print an ASCII diagram of the flow.
    """
    diagram = r"""
Legend:
  ❄️  frozen (not trainable)   🔧  trainable

Raw video:
  images: [B, T, C, H, W]
      │
      ▼
Vision encoder (InternViT) ❄️:
  vit_tokens: [B*T, N_patch, D]
      │          (tokens over patches for each frame; encoder weights frozen)
      │  mean over patches (dim=1)
      ▼
  frame_tokens: [B, T, D]
      │          (one token per frame; still treated as frozen features)
      │  concatenate with text tokens along sequence dim
      ▼
LLM input:
  [frame_tokens | text_tokens] -> [B, T + L_text, D]
      │
      ▼
LLM (Qwen3) ❄️ hidden states (last layer):
  hidden: [B, T + L_text, D]
      │
      └─ take first T tokens (image segment) -> img_hidden: [B, T, D]
                         │
                         │  mean over frames (dim=1) 🔧
                         ▼
Final feature to classifier head 🔧:
  pooled_feats: [B, D]
"""
    print(diagram)


if __name__ == "__main__":
    mean_pooling_flow_demo()
    print_ascii_diagram()


