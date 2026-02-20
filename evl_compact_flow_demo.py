import torch


def evl_compact_flow_demo(
    batch_size: int = 2,
    num_frames: int = 32,
    num_patches: int = 576,
    hidden_dim: int = 1024,
    num_queries: int = 8,
):
    """
    Demonstrate (with shapes only) how the EVL compact video experiment processes
    images from patches -> frame tokens -> EVL decoder queries q_M -> final mean
    over queries. This mirrors the flow used in `train_evl_compact_classifier.py`.
    """
    B = batch_size
    T = num_frames
    N = num_patches
    D = hidden_dim
    M = num_queries

    print("Legend: ❄️ frozen module (not trainable), 🔧 trainable module\n")

    print("=== Step 1: Raw video batch ===")
    images = torch.randn(B, T, 3, 224, 224)
    print(f"images:           {tuple(images.shape)}  [B, T, C, H, W]")

    print("\n=== Step 2: Flatten time for vision encoder ===")
    images_bt = images.view(B * T, 3, 224, 224)
    print(f"images_bt:        {tuple(images_bt.shape)}  [B*T, C, H, W]")

    print("\n=== Step 3: Vision encoder outputs patch tokens ===")
    # In the real experiment, InternVL's frozen vision encoder (InternViT) produces
    # one token per image patch. Here we simulate that with random tensors.
    vit_tokens = torch.randn(B * T, N, D)
    print(f"vit_tokens:       {tuple(vit_tokens.shape)}  [B*T, N_patch, D]")

    print("\n=== Step 4: Mean over patches -> one token per frame ===")
    frame_tokens_bt = vit_tokens.mean(dim=1)  # collapse patches
    print(f"frame_tokens_bt:  {tuple(frame_tokens_bt.shape)}  [B*T, D]")
    frame_tokens = frame_tokens_bt.view(B, T, D)
    print(f"frame_tokens:     {tuple(frame_tokens.shape)}  [B, T, D] (one token per frame)")

    print("\n=== Step 5: Add temporal position encoding (memory) ===")
    pos = torch.randn(T, D)  # stand-in for learnable temporal_pos_emb
    tokens_with_pos = frame_tokens + pos.unsqueeze(0)
    print(f"tokens_with_pos:  {tuple(tokens_with_pos.shape)}  [B, T, D]")

    print("\n=== Step 6: EVL temporal decoder with queries q_M ===")
    # In the real code, query_tokens are learnable parameters (trainable).
    query_tokens = torch.randn(M, D)
    print(f"query_tokens 🔧:  {tuple(query_tokens.shape)}  [M, D] (learnable)")

    # Expand queries for the batch
    queries = query_tokens.unsqueeze(0).expand(B, -1, -1)
    print(f"queries:          {tuple(queries.shape)}  [B, M, D]")

    # The actual TransformerDecoder performs attention from queries to memory.
    # Here we simulate its output with random tensors of the same shape.
    compact_tokens = torch.randn(B, M, D)
    print(f"compact_tokens 🔧:{tuple(compact_tokens.shape)}  [B, M, D] (q_M after decoder)")

    print("\n=== Step 7: Mean over queries -> video feature ===")
    pooled_feats = compact_tokens.mean(dim=1)
    print(f"pooled_feats 🔧:  {tuple(pooled_feats.shape)}  [B, D] (mean over M queries)")

    print("\n=== Summary (collapses) ===")
    print("1) Vision encoder ❄️: [B, T, C, H, W] -> [B*T, N_patch, D]  (frozen InternViT)")
    print("2) Spatial mean:       [B*T, N_patch, D] --mean over patches--> [B, T, D]")
    print("3) EVL decoder 🔧:     [B, T, D] (memory with pos) + M queries -> compact_tokens [B, M, D]")
    print("4) Query mean 🔧:      [B, M, D] --mean over queries--> [B, D]  (feature fed to trainable classifier)")

    print("\nKey difference vs mean-pooling experiment:")
    print("- Here we do NOT directly average over frames T.")
    print("- Temporal structure is encoded inside the EVL decoder attention,")
    print("  and we only average over the M compact queries, not over time indices.")


def print_evl_ascii_diagram():
    """
    Optional: print an ASCII diagram of the EVL compact-token flow.
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
Frame tokens:
  frame_tokens: [B, T, D]
      │
      │  add temporal position encoding (learnable, part of EVL adapter) 🔧
      ▼
Memory for EVL decoder:
  memory: [B, T, D]
      │
      ├─ Learnable queries q_M 🔧: [M, D]
      │         │
      │         ▼
      │    expanded to batch: [B, M, D]
      │
      ▼
EVL temporal decoder 🔧 (TransformerDecoder):
  compact_tokens (q_M): [B, M, D]
      │
      │  mean over queries (dim=1) 🔧
      ▼
Final feature to classifier head 🔧:
  pooled_feats: [B, D]
"""
    print(diagram)


if __name__ == "__main__":
    evl_compact_flow_demo()
    print_evl_ascii_diagram()















