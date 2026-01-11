import torch
import torch.nn as nn


class TemporalVideoEncoder(nn.Module):
    """
    Simple EVL-style *encoder* adapter on top of frozen InternVL vision features.

    - Uses the frozen InternVL vision backbone via `base_model.extract_feature`
      to obtain one feature vector per frame.
    - Optionally fuses per-frame skeleton features with the image feature.
    - Applies a small Transformer encoder over time to get a video-level feature.
    - Only the skeleton projection + temporal encoder are trainable; the
      InternVL backbone stays frozen.
    """

    def __init__(
        self,
        base_model: nn.Module,
        max_frames: int = 32,
        skel_dim: int = 46,
        use_skeleton: bool = True,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        # Freeze all original parameters
        self.base_model.requires_grad_(False)

        # Hidden size is taken from the language model config if present
        language_model = getattr(self.base_model, "language_model", None)
        if language_model is not None:
            hidden_size = language_model.config.hidden_size
            try:
                lm_dtype = next(language_model.parameters()).dtype
            except StopIteration:
                lm_dtype = torch.float32
        else:
            hidden_size = self.base_model.config.hidden_size
            try:
                lm_dtype = next(self.base_model.parameters()).dtype
            except StopIteration:
                lm_dtype = torch.float32

        self.hidden_size = hidden_size
        self.max_frames = max_frames
        self.skel_dim = skel_dim
        self.use_skeleton = use_skeleton

        # Optional skeleton projection: 46-dim -> hidden_size per frame
        if use_skeleton:
            self.skel_proj = nn.Linear(skel_dim, hidden_size, dtype=lm_dtype)
            self.skel_ln = nn.LayerNorm(hidden_size, eps=1e-5, dtype=lm_dtype)

        # Learnable temporal positional embeddings for up to max_frames
        self.temporal_pos_emb = nn.Parameter(torch.zeros(max_frames, hidden_size, dtype=lm_dtype))

        # Lightweight Transformer encoder over the temporal dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of frames into one visual token per frame using the
        frozen InternVL vision backbone.

        Args:
            pixel_values: [B, T, 3, H, W] tensor

        Returns:
            img_tokens: [B, T, hidden_size]
        """
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B, T, 3, H, W], got {pixel_values.shape}")

        device = next(self.parameters()).device

        # Match dtype with the frozen model parameters
        try:
            dtype = next(self.base_model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

        B, T, C, H, W = pixel_values.shape
        if T > self.max_frames:
            raise ValueError(f"T = {T} exceeds max_frames = {self.max_frames}")

        flat = pixel_values.to(device=device, dtype=dtype).view(B * T, C, H, W)  # [B*T, 3, H, W]

        # Use InternVL's custom vision encoder API
        with torch.no_grad():
            vit_embeds = self.base_model.extract_feature(flat)  # [B*T, N_tokens, D]

        # Pool spatial tokens to a single token per frame (mean over patches)
        frame_tokens = vit_embeds.mean(dim=1)  # [B*T, D]
        img_tokens = frame_tokens.view(B, T, -1)  # [B, T, D]
        return img_tokens

    def forward(
        self,
        pixel_values: torch.Tensor,
        skeleton_feats: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, T, 3, H, W]
            skeleton_feats: [B, T, skel_dim] or None

        Returns:
            video_features: [B, hidden_size] video-level representation
        """
        device = next(self.parameters()).device

        img_tokens = self.encode_images(pixel_values)  # [B, T, D]
        B, T, D = img_tokens.shape

        fused = img_tokens
        if self.use_skeleton:
            if skeleton_feats is None:
                raise ValueError("skeleton_feats must be provided when use_skeleton=True")
            if skeleton_feats.ndim != 3:
                raise ValueError(
                    f"skeleton_feats must be [B, T, {self.skel_dim}], got {skeleton_feats.shape}"
                )
            if skeleton_feats.shape[1] != T:
                raise ValueError(
                    f"skeleton_feats length {skeleton_feats.shape[1]} must match image frames {T}"
                )
            if skeleton_feats.shape[2] != self.skel_dim:
                raise ValueError(
                    f"Expected skeleton dim {self.skel_dim}, got {skeleton_feats.shape[2]}"
                )

            target_dtype = self.temporal_pos_emb.dtype
            skel_feats = skeleton_feats.to(device=device, dtype=target_dtype)

            skel_tokens = self.skel_proj(skel_feats)  # [B, T, D]
            skel_tokens = self.skel_ln(skel_tokens)

            img_tokens = img_tokens.to(device=device, dtype=target_dtype)
            fused = img_tokens + skel_tokens  # simple fusion at feature level

        # Add temporal positional encoding
        pos = self.temporal_pos_emb[:T].unsqueeze(0).to(device)  # [1, T, D]
        tokens = fused + pos  # [B, T, D]

        # Temporal Transformer encoder over the sequence of frame tokens
        tokens = self.temporal_encoder(tokens)  # [B, T, D]

        # Simple video-level pooling (mean over time)
        video_features = tokens.mean(dim=1)  # [B, D]
        return video_features


class EVLTemporalDecoder(nn.Module):
    """
    EVL-style *decoder* adapter that produces a small set of compact video
    tokens q_M from per-frame InternVL vision features.

    - Vision backbone (InternVL's ViT) is frozen.
    - A learnable set of M queries attends to all frame tokens through a stack
      of TransformerDecoder layers.
    - The resulting compact tokens can be (1) fed directly to a classifier or
      (2) used as visual prefix tokens for the LLM.
    """

    def __init__(
        self,
        base_model: nn.Module,
        max_frames: int = 32,
        num_queries: int = 8,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.base_model.requires_grad_(False)

        language_model = getattr(self.base_model, "language_model", None)
        if language_model is not None:
            hidden_size = language_model.config.hidden_size
            try:
                lm_dtype = next(language_model.parameters()).dtype
            except StopIteration:
                lm_dtype = torch.float32
        else:
            hidden_size = self.base_model.config.hidden_size
            try:
                lm_dtype = next(self.base_model.parameters()).dtype
            except StopIteration:
                lm_dtype = torch.float32

        self.hidden_size = hidden_size
        self.max_frames = max_frames
        self.num_queries = num_queries

        # Learnable query tokens q_0 ... q_M-1
        self.query_tokens = nn.Parameter(
            torch.zeros(num_queries, hidden_size, dtype=lm_dtype)
        )

        # Optional temporal positional encoding for the memory (frame tokens)
        self.temporal_pos_emb = nn.Parameter(
            torch.zeros(max_frames, hidden_size, dtype=lm_dtype)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode frames into one token per frame using the frozen vision backbone.

        Args:
            pixel_values: [B, T, 3, H, W]

        Returns:
            frame_tokens: [B, T, D]
        """
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B, T, 3, H, W], got {pixel_values.shape}")

        device = next(self.parameters()).device

        try:
            dtype = next(self.base_model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

        B, T, C, H, W = pixel_values.shape
        if T > self.max_frames:
            raise ValueError(f"T = {T} exceeds max_frames = {self.max_frames}")

        flat = pixel_values.to(device=device, dtype=dtype).view(B * T, C, H, W)
        with torch.no_grad():
            vit_embeds = self.base_model.extract_feature(flat)  # [B*T, N_tokens, D]

        frame_tokens = vit_embeds.mean(dim=1)  # [B*T, D]
        frame_tokens = frame_tokens.view(B, T, -1)  # [B, T, D]
        return frame_tokens

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, T, 3, H, W]

        Returns:
            compact_tokens: [B, num_queries, hidden_size]  (q_M in the diagram)
        """
        device = next(self.parameters()).device
        memory = self.encode_images(pixel_values)  # [B, T, D]
        B, T, D = memory.shape

        # Add temporal position encoding to memory
        pos = self.temporal_pos_emb[:T].unsqueeze(0).to(device)  # [1, T, D]
        memory = memory + pos

        # Expand learnable queries for the batch
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1).to(device)  # [B, M, D]

        compact_tokens = self.temporal_decoder(tgt=queries, memory=memory)  # [B, M, D]
        return compact_tokens

