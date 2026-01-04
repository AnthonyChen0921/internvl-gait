import torch
import torch.nn as nn
from typing import Optional
from transformers.modeling_outputs import BaseModelOutput


class InternVLWithSkeleton(nn.Module):
    """
    Wrapper around a pretrained InternVL model that adds a skeleton-token pathway.

    - Keeps the original InternVL weights frozen.
    - Adds a small projection + positional embedding for 46-dim skeleton features.
    - Fuses skeleton tokens with text (and later optionally vision) at the LLM input.
    """

    def __init__(self, base_model: nn.Module, max_frames: int = 100, skel_dim: int = 46):
        super().__init__()
        self.base_model = base_model
        # Freeze all original parameters
        self.base_model.requires_grad_(False)

        # Underlying language model (Qwen3 inside InternVL)
        self.language_model = getattr(self.base_model, "language_model", self.base_model)
        hidden_size = self.language_model.config.hidden_size

        # Match dtype with the language model (now float32) to avoid casting issues
        try:
            lm_dtype = next(self.language_model.parameters()).dtype
        except StopIteration:
            lm_dtype = torch.float32

        self.hidden_size = hidden_size
        self.max_frames = max_frames
        self.skel_dim = skel_dim

        # 46-dim skeleton -> hidden_size token projection
        self.skel_proj = nn.Linear(skel_dim, hidden_size, dtype=lm_dtype)
        # Learnable temporal embeddings for up to max_frames
        self.skel_pos_emb = nn.Parameter(torch.zeros(max_frames, hidden_size, dtype=lm_dtype))
        self.skel_ln = nn.LayerNorm(hidden_size, eps=1e-5, dtype=lm_dtype)

    def encode_skeleton(self, skeleton_feats: torch.Tensor) -> torch.Tensor:
        """
        Encode skeleton time series into token embeddings.

        Args:
            skeleton_feats: [B, T, skel_dim] float tensor

        Returns:
            skel_tokens: [B, T, hidden_size]
        """
        if skeleton_feats.ndim != 3:
            raise ValueError(f"skeleton_feats must be [B, T, {self.skel_dim}], got {skeleton_feats.shape}")

        B, T, C = skeleton_feats.shape
        if C != self.skel_dim:
            raise ValueError(f"Expected last dim {self.skel_dim}, got {C}")
        if T > self.max_frames:
            raise ValueError(f"T = {T} exceeds max_frames = {self.max_frames}")

        device = next(self.parameters()).device
        target_dtype = self.skel_pos_emb.dtype
        skeleton_feats = skeleton_feats.to(device=device, dtype=target_dtype)

        proj = self.skel_proj(skeleton_feats)  # [B, T, D]
        pos = self.skel_pos_emb[:T].unsqueeze(0).to(device)  # [1, T, D]
        tokens = proj + pos
        tokens = self.skel_ln(tokens)
        return tokens

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of frames into one visual token per frame.

        Args:
            pixel_values: [B, T, 3, H, W] tensor

        Returns:
            img_tokens: [B, T, hidden_size]
        """
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B, T, 3, H, W], got {pixel_values.shape}")

        device = next(self.parameters()).device
        dtype = next(self.language_model.parameters()).dtype

        B, T, C, H, W = pixel_values.shape
        flat = pixel_values.to(device=device, dtype=dtype).view(B * T, C, H, W)  # [B*T, 3, H, W]

        # Use base_model's vision encoder via extract_feature (InternVL custom API).
        with torch.no_grad():
            vit_embeds = self.base_model.extract_feature(flat)  # [B*T, N_tokens, D]

        # Pool spatial tokens to a single token per frame (mean over patches).
        frame_tokens = vit_embeds.mean(dim=1)  # [B*T, D]
        img_tokens = frame_tokens.view(B, T, -1)  # [B, T, D]
        return img_tokens

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        skeleton_feats: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **lm_kwargs,
    ):
        """
        Forward pass that optionally prepends image and/or skeleton tokens before
        text tokens and feeds the fused sequence into the underlying language model.

        This is suitable for training (returns standard LM outputs).
        """
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Text embeddings from the language model's input embedding layer
        text_embeds = self.language_model.get_input_embeddings()(input_ids)  # [B, L, D]

        prefix_tokens = []
        prefix_masks = []

        # Image tokens (optional)
        if pixel_values is not None:
            img_tokens = self.encode_images(pixel_values)  # [B, T_img, D]
            B, T_img, _ = img_tokens.shape
            img_mask = torch.ones(
                B,
                T_img,
                dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=device,
            )
            prefix_tokens.append(img_tokens)
            prefix_masks.append(img_mask)

        # Skeleton tokens (optional)
        if skeleton_feats is not None:
            skel_tokens = self.encode_skeleton(skeleton_feats)  # [B, T_skel, D]
            B, T_skel, _ = skel_tokens.shape
            skel_mask = torch.ones(
                B,
                T_skel,
                dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=device,
            )
            prefix_tokens.append(skel_tokens)
            prefix_masks.append(skel_mask)

        if prefix_tokens:
            prefix_tokens_cat = torch.cat(prefix_tokens, dim=1)  # [B, T_prefix, D]
            prefix_masks_cat = torch.cat(prefix_masks, dim=1)    # [B, T_prefix]

            inputs_embeds = torch.cat([prefix_tokens_cat, text_embeds], dim=1)  # [B, T_prefix+L, D]
            if attention_mask is None:
                fused_mask = prefix_masks_cat
            else:
                fused_mask = torch.cat([prefix_masks_cat, attention_mask], dim=1)  # [B, T_prefix+L]

            return self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=fused_mask,
                **lm_kwargs,
            )
        else:
            # Fall back to standard text-only behavior
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **lm_kwargs,
            )

    @torch.no_grad()
    def generate_with_skeleton(
        self,
        tokenizer,
        prompt: str,
        skeleton_feats: torch.Tensor,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        **generate_kwargs,
    ):
        """
        Convenience helper for inference:
        - tokenizes the text prompt
        - prepends skeleton tokens
        - calls the underlying language_model.generate with inputs_embeds
        """
        device = next(self.parameters()).device

        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)

        text_embeds = self.language_model.get_input_embeddings()(input_ids)  # [B, L, D]
        skel_tokens = self.encode_skeleton(skeleton_feats.to(device))  # [B, T, D]

        B, T, _ = skel_tokens.shape
        if attention_mask is None:
            skel_mask = torch.ones(B, T, dtype=torch.long, device=device)
            fused_mask = skel_mask
        else:
            skel_mask = torch.ones(B, T, dtype=attention_mask.dtype, device=device)
            fused_mask = torch.cat([skel_mask, attention_mask], dim=1)  # [B, T+L]

        inputs_embeds = torch.cat([skel_tokens, text_embeds], dim=1)  # [B, T+L, D]

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=fused_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **generate_kwargs,
        )
        return outputs


class InternVLWithSkeletonXAttn(nn.Module):
    """
    Variant that uses a cross-attention adapter:
      - Text tokens are processed by the frozen InternVL language model.
      - Skeleton tokens are encoded separately and used as key/value in a
        trainable cross-attention block.
      - The adapter output (text states enhanced by skeleton) is then used for
        classification.
    """

    def __init__(self, base_model: nn.Module, max_frames: int = 100, skel_dim: int = 46, num_heads: int = 8):
        super().__init__()
        self.base_model = base_model
        self.base_model.requires_grad_(False)

        self.language_model = getattr(self.base_model, "language_model", self.base_model)
        hidden_size = self.language_model.config.hidden_size

        try:
            lm_dtype = next(self.language_model.parameters()).dtype
        except StopIteration:
            lm_dtype = torch.float32

        self.hidden_size = hidden_size
        self.max_frames = max_frames
        self.skel_dim = skel_dim

        # Skeleton encoder (same as InternVLWithSkeleton)
        self.skel_proj = nn.Linear(skel_dim, hidden_size, dtype=lm_dtype)
        self.skel_pos_emb = nn.Parameter(torch.zeros(max_frames, hidden_size, dtype=lm_dtype))
        self.skel_ln = nn.LayerNorm(hidden_size, eps=1e-5, dtype=lm_dtype)

        # Cross-attention block: queries = text tokens, keys/values = skeleton tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,  # so we can use [B, L, D] directly
            dtype=lm_dtype,
        )
        self.cross_ln = nn.LayerNorm(hidden_size, eps=1e-5, dtype=lm_dtype)

    def encode_skeleton(self, skeleton_feats: torch.Tensor) -> torch.Tensor:
        """
        Encode skeleton time series into token embeddings [B, T, D].
        """
        if skeleton_feats.ndim != 3:
            raise ValueError(f"skeleton_feats must be [B, T, {self.skel_dim}], got {skeleton_feats.shape}")

        B, T, C = skeleton_feats.shape
        if C != self.skel_dim:
            raise ValueError(f"Expected last dim {self.skel_dim}, got {C}")
        if T > self.max_frames:
            raise ValueError(f"T = {T} exceeds max_frames = {self.max_frames}")

        device = next(self.parameters()).device
        target_dtype = self.skel_pos_emb.dtype
        skeleton_feats = skeleton_feats.to(device=device, dtype=target_dtype)

        proj = self.skel_proj(skeleton_feats)  # [B, T, D]
        pos = self.skel_pos_emb[:T].unsqueeze(0).to(device)  # [1, T, D]
        tokens = proj + pos
        tokens = self.skel_ln(tokens)
        return tokens

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        skeleton_feats: Optional[torch.FloatTensor] = None,
        **lm_kwargs,
    ):
        """
        Forward pass:
          1) Get text embeddings and run frozen LM to get last_hidden_state.
          2) If skeleton_feats is provided, encode them and run a cross-attn
             block where text (LM output) attends to skeleton tokens.
          3) Return a BaseModelOutput with last_hidden_state enhanced by skeleton.
        """
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Text embeddings and frozen LM
        text_embeds = self.language_model.get_input_embeddings()(input_ids)  # [B, L, D]

        # Ensure we get hidden_states back, but don't override explicit kwargs
        if "output_hidden_states" not in lm_kwargs:
            lm_kwargs["output_hidden_states"] = True
        if "return_dict" not in lm_kwargs:
            lm_kwargs["return_dict"] = True

        lm_outputs = self.language_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            **lm_kwargs,
        )
        # Use the last layer hidden states from the LM
        hidden = lm_outputs.hidden_states[-1]  # [B, L, D]

        if skeleton_feats is not None:
            skel_tokens = self.encode_skeleton(skeleton_feats)  # [B, T, D]
            # Cross-attention: queries = hidden (text), keys/values = skeleton
            attn_out, _ = self.cross_attn(
                query=hidden,       # [B, L, D]
                key=skel_tokens,    # [B, T, D]
                value=skel_tokens,  # [B, T, D]
                need_weights=False,
            )
            hidden = self.cross_ln(hidden + attn_out)  # residual + LN

        return BaseModelOutput(last_hidden_state=hidden)


