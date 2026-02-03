import torch


def frame_aligned_evl_flow_demo(
    batch_size: int = 2,
    num_frames: int = 32,
    num_patches: int = 576,
    hidden_dim: int = 1024,
    text_len: int = 200,
):
    """
    Demonstrate (with shapes only) how the FrameAlignedEVL decoder keeps
    frame alignment while adding temporal aggregation. This mirrors the flow
    used in `train_evl_framealigned_skeltext_classifier.py`.
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
    vit_tokens = torch.randn(B * T, N, D)
    print(f"vit_tokens:       {tuple(vit_tokens.shape)}  [B*T, N_patch, D]")

    print("\n=== Step 4: Mean over patches -> one token per frame ===")
    frame_tokens_bt = vit_tokens.mean(dim=1)
    print(f"frame_tokens_bt:  {tuple(frame_tokens_bt.shape)}  [B*T, D]")
    frame_tokens = frame_tokens_bt.view(B, T, D)
    print(f"frame_tokens:     {tuple(frame_tokens.shape)}  [B, T, D] (one token per frame)")

    print("\n=== Step 5: Add temporal position encoding ===")
    pos = torch.randn(T, D)
    memory = frame_tokens + pos.unsqueeze(0)
    print(f"memory:           {tuple(memory.shape)}  [B, T, D]")

    print("\n=== Step 6: Frame-aligned EVL decoder ===")
    # Decoder uses tgt = frame_tokens (+pos) and memory = frame_tokens (+pos)
    decoded = torch.randn(B, T, D)
    print(f"decoded 🔧:       {tuple(decoded.shape)}  [B, T, D]")

    print("\n=== Step 7: Residual to preserve alignment ===")
    frame_aligned = frame_tokens + decoded
    print(f"frame_aligned 🔧: {tuple(frame_aligned.shape)}  [B, T, D] (aligned to frames)")

    print("\n=== Step 8: Concatenate with text tokens and run LLM ===")
    text_tokens = torch.randn(B, L, D)
    print(f"text_tokens:      {tuple(text_tokens.shape)}  [B, L_text, D]")
    llm_input = torch.cat([frame_aligned, text_tokens], dim=1)
    print(f"llm_input:        {tuple(llm_input.shape)}  [B, T + L_text, D]")

    print("\n=== Step 9: LLM output ===")
    hidden = llm_input
    print(f"hidden:           {tuple(hidden.shape)}  [B, T + L_text, D]")

    print("\n=== Step 10: Mean over frames ===")
    img_hidden = hidden[:, :T, :]
    pooled_feats = img_hidden.mean(dim=1)
    print(f"pooled_feats 🔧:  {tuple(pooled_feats.shape)}  [B, D] (mean over frames)")

    print("\n=== Summary (collapses) ===")
    print("1) Vision encoder ❄️: [B, T, C, H, W] -> [B*T, N_patch, D]")
    print("2) Spatial mean:       [B*T, N_patch, D] --mean over patches--> [B, T, D]")
    print("3) Frame-aligned EVL 🔧: Decoder(tgt=frame_tokens, memory=frame_tokens) + residual -> [B, T, D]")
    print("4) LLM ❄️:            [B, T + L_text, D] -> take first T tokens -> [B, T, D]")
    print("5) Temporal mean 🔧:  [B, T, D] --mean over frames--> [B, D]")


def print_frame_aligned_ascii_diagram():
    diagram = r"""
Legend:
  ❄️  frozen (not trainable)   🔧  trainable

Raw video:
  images: [B, T, C, H, W]
      │
      ▼
Vision encoder (InternViT) ❄️:
  vit_tokens: [B*T, N_patch, D]
      │
      │  mean over patches (dim=1)
      ▼
Frame tokens:
  frame_tokens: [B, T, D]
      │
      │  add temporal position encoding 🔧
      ▼
Frame-aligned EVL decoder 🔧:
  decoded: [B, T, D]
      │
      │  residual add (preserve alignment)
      ▼
frame_aligned: [B, T, D]
      │
      │  concatenate with text tokens
      ▼
LLM input:
  [frame_aligned | text_tokens] -> [B, T + L_text, D]
      │
      ▼
LLM hidden ❄️:
  hidden: [B, T + L_text, D]
      │
      └─ take first T tokens -> [B, T, D]
                         │
                         │  mean over frames (dim=1) 🔧
                         ▼
Final feature to classifier head 🔧:
  pooled_feats: [B, D]
"""
    print(diagram)


if __name__ == "__main__":
    frame_aligned_evl_flow_demo()
    print_frame_aligned_ascii_diagram()
