import torch


def evl_compact_text_flow_demo(
    batch_size: int = 2,
    num_frames: int = 32,
    num_patches: int = 576,
    hidden_dim: int = 1024,
    num_queries: int = 8,
    text_len: int = 200,
):
    """
    Demonstrate (with shapes only) how the EVL compact + LLM experiment processes
    images from patches -> frame tokens (memory) -> EVL decoder queries q_M ->
    LLM with text prompt -> final pooled video feature.

    This mirrors the flow in `train_evl_compact_text_classifier.py`:
      - Frozen vision encoder (InternViT) ❄️
      - EVLTemporalDecoder + query tokens 🔧
      - Frozen language model (Qwen3) ❄️
      - Trainable classifier head 🔧
    """
    B = batch_size
    T = num_frames
    N = num_patches
    D = hidden_dim
    M = num_queries
    L = text_len

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
    print(f"vit_tokens ❄️:    {tuple(vit_tokens.shape)}  [B*T, N_patch, D]  (frozen InternViT)")

    print("\n=== Step 4: Mean over patches -> one token per frame (memory) ===")
    frame_tokens_bt = vit_tokens.mean(dim=1)  # collapse patches
    print(f"frame_tokens_bt:  {tuple(frame_tokens_bt.shape)}  [B*T, D]")
    frame_tokens = frame_tokens_bt.view(B, T, D)
    print(f"frame_tokens:     {tuple(frame_tokens.shape)}  [B, T, D]  (one token per frame)")

    print("\n=== Step 5: Add temporal position encoding (EVL memory) ===")
    pos = torch.randn(T, D)  # stand-in for learnable temporal_pos_emb 🔧
    memory = frame_tokens + pos.unsqueeze(0)
    print(f"memory 🔧:        {tuple(memory.shape)}  [B, T, D]  (frame tokens + temporal PE)")

    print("\n=== Step 6: EVL temporal decoder with learnable queries q_M ===")
    query_tokens = torch.randn(M, D)
    print(f"query_tokens 🔧:  {tuple(query_tokens.shape)}  [M, D]  (learnable)")

    # Expand queries for the batch
    queries = query_tokens.unsqueeze(0).expand(B, -1, -1)
    print(f"queries:          {tuple(queries.shape)}  [B, M, D]")

    # The actual TransformerDecoder performs attention from queries to memory.
    # Here we simulate its output with random tensors of the same shape.
    compact_tokens = torch.randn(B, M, D)
    print(f"compact_tokens 🔧:{tuple(compact_tokens.shape)}  [B, M, D]  (q_M after EVL decoder)")

    print("\n=== Step 7: Build LLM inputs: [q_M | text] ===")
    text_embeds = torch.randn(B, L, D)
    print(f"text_embeds ❄️:  {tuple(text_embeds.shape)}  [B, L, D]  (frozen token embeddings from LLM)")

    inputs_embeds = torch.cat([compact_tokens, text_embeds], dim=1)
    print(f"inputs_embeds:    {tuple(inputs_embeds.shape)}  [B, M + L, D]")

    print("\n=== Step 8: LLM processes [q_M | text] ===")
    # In the real code, this is the last hidden state from the frozen LLM.
    # Here we simulate with a simple identity to keep the demo light-weight.
    hidden = inputs_embeds
    print(f"hidden ❄️:        {tuple(hidden.shape)}  [B, M + L, D]  (LLM last hidden states)")

    print("\n=== Step 9: Take video part and pool over queries ===")
    video_hidden = hidden[:, :M, :]
    print(f"video_hidden:     {tuple(video_hidden.shape)}  [B, M, D]  (q_M after LLM)")

    pooled_feats = video_hidden.mean(dim=1)
    print(f"pooled_feats 🔧:  {tuple(pooled_feats.shape)}  [B, D]  (mean over M, fed to classifier)")

    print("\n=== Summary (EVL + LLM) ===")
    print("1) Vision encoder ❄️: [B, T, C, H, W] -> [B*T, N_patch, D]  (frozen InternViT)")
    print("2) Spatial mean:       [B*T, N_patch, D] --mean over patches--> [B, T, D]")
    print("3) EVL memory 🔧:      [B, T, D] + temporal PE -> memory [B, T, D]")
    print("4) EVL decoder 🔧:     memory [B, T, D] + queries [M, D] -> q_M [B, M, D]")
    print("5) LLM input ❄️:      [q_M | text] -> inputs_embeds [B, M+L, D]")
    print("6) LLM hidden ❄️:     hidden [B, M+L, D] -> take first M tokens -> video_hidden [B, M, D]")
    print("7) Query mean 🔧:      video_hidden [B, M, D] --mean over queries--> [B, D] (classifier input)")

    print("\nKey points:")
    print("- Vision encoder and LLM weights are frozen ❄️.")
    print("- Only EVL adapter (temporal PE + decoder + queries) and classifier are trainable 🔧.")
    print("- Temporal structure is modeled in the EVL decoder; the LLM refines q_M in a text-aware space.")


def print_evl_llm_ascii_diagram():
    """
    Optional: print an ASCII diagram of the EVL compact + LLM flow.
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
      │  add temporal position encoding 🔧
      ▼
EVL memory 🔧:
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
      │  concatenate with text embeddings from LLM ❄️
      ▼
LLM input ❄️:
  inputs_embeds: [q_M | text] -> [B, M + L, D]
      │
      ▼
Frozen LLM (Qwen3) ❄️:
  hidden: [B, M + L, D]
      │
      └─ take first M tokens (video segment) -> video_hidden: [B, M, D]
                         │
                         │  mean over queries (dim=1) 🔧
                         ▼
Final feature to classifier head 🔧:
  pooled_feats: [B, D]
"""
    print(diagram)


if __name__ == "__main__":
    evl_compact_text_flow_demo()
    print_evl_llm_ascii_diagram()















