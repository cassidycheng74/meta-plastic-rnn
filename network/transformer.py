"""
Design:
    - Causal (masked) self-attention: output at timestep t depends only
      on inputs up to t, matching the RNN's causal structure 
    - Same input/output interface as LeakyRNN and GRUNet: takes
      (T, B, n_input) and returns NetworkOutput with fields
      output, hidden, loss_reg.
    - Sinusoidal positional encoding so the network knows where it is
      in the trial timeline.
    - No cross-trial state: hidden state resets between trials, same
      as the RNN implementations.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from typing import Optional

from network.rnn import NetworkOutput

# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds a fixed (non-learned) position signal to the input embeddings
    so the transformer knows where each timestep falls in the trial.

    """

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Shape: (max_len, 1, d_model) for broadcasting over batch.
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, d_model)
        Returns:
            x + positional encoding, same shape
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# ---------------------------------------------------------------------------
# TransformerNet
# ---------------------------------------------------------------------------

class TransformerNet(nn.Module):
    """
    Args:
        n_input:    number of input channels (11 in the unified format)
        n_output:   number of output channels (5)
        d_model:    internal embedding dimension
        n_heads:    number of attention heads (must divide d_model)
        n_layers:   number of transformer encoder layers
        d_ff:       feedforward layer dimension (typically 2-4x d_model)
        dropout:    dropout rate applied in attention and feedforward layers
        max_len:    max trial length in timesteps for positional encoding
        l2_weight:  L2 regularization on all parameters
    """

    def __init__(
        self,
        n_input:   int,
        n_output:  int,
        d_model:   int   = 128,
        n_heads:   int   = 4,
        n_layers:  int   = 3,
        d_ff:      int   = 256,
        dropout:   float = 0.1,
        max_len:   int   = 2000,
        l2_weight: float = 1e-6,
    ):
        super().__init__()

        assert d_model % n_heads == 0, (
            f'd_model ({d_model}) must be divisible by n_heads ({n_heads})')

        self.d_model   = d_model
        self.n_input   = n_input
        self.n_output  = n_output
        self.l2_weight = l2_weight

        # --- Input projection ---
        # Maps the 11-channel input to the transformer's internal dimension.
        self.input_proj = nn.Linear(n_input, d_model)

        # --- Positional encoding ---
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model = d_model,
            max_len = max_len,
            dropout = dropout,
        )

        # --- Transformer encoder ---
        # Each layer: causal self-attention + feedforward.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = False,   # expects (T, B, d_model)
            norm_first      = True,    # pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers    = n_layers,
            # Final layer norm for stable output.
            norm          = nn.LayerNorm(d_model),
        )

        # --- Output projection ---
        # Maps d_model back to n_output at every timestep.
        self.output_proj = nn.Linear(d_model, n_output)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        nn.init.normal_(self.input_proj.weight,  std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

        # Initialize transformer layers.
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal attention mask.

        Entry (i, j) is True if position i should NOT attend to position j.
        Upper triangle = future positions = masked out.

        Shape: (T, T)
        """
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1
        )

    def init_hidden(self, batch_size: int, device: torch.device):
        """
        API compatibility with LeakyRNN — transformers have no hidden state.
        Returns None; the forward pass ignores it.
        """
        return None

    def forward(
        self,
        x:  torch.Tensor,            # (T, B, n_input)
        h0: Optional[object] = None, # ignored — no recurrent state
    ) -> NetworkOutput:
        """
        Forward pass over a full trial sequence.

        Args:
            x:  input tensor (T, B, n_input)
            h0: ignored (API compatibility with RNN interface)

        Returns:
            NetworkOutput with:
                output:   (T, B, n_output) — predictions at every timestep
                hidden:   (T, B, d_model)  — transformer hidden states
                loss_reg: scalar           — L2 regularization loss
        """
        T, B, _ = x.shape
        device  = x.device

        # Project inputs to embedding dimension.
        h = self.input_proj(x)        # (T, B, d_model)

        # Add positional encoding.
        h = self.pos_enc(h)           # (T, B, d_model)

        # Build causal attention mask.
        causal_mask = self._causal_mask(T, device)

        # Transformer forward with causal masking.
        h = self.transformer(h, mask=causal_mask)   # (T, B, d_model)

        # Project to output space.
        output = self.output_proj(h)  # (T, B, n_output)

        loss_reg = self._regularization_loss()

        return NetworkOutput(output=output, hidden=h, loss_reg=loss_reg)

    def _regularization_loss(self) -> torch.Tensor:
        """L2 regularization on all trainable parameters."""
        if self.l2_weight <= 0:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for p in self.parameters():
            if p.requires_grad:
                loss = loss + p.pow(2).mean()
        return self.l2_weight * loss

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Attention extraction utility (for Phase 7 analysis)
# ---------------------------------------------------------------------------

class TransformerNetWithAttention(TransformerNet):
    """
    Extended version that returns attention weights for analysis.

    Usage:
        model = TransformerNetWithAttention(...)
        output, hidden, attn_weights = model.forward_with_attention(x)
        # attn_weights: list of (B, n_heads, T, T) tensors, one per layer
    """

    def forward_with_attention(self, x: torch.Tensor):
        """
        Forward pass that also returns attention weights.

        Returns:
            output:       (T, B, n_output)
            hidden:       (T, B, d_model)
            attn_weights: list of (B, n_heads, T, T), one per layer
        """
        T, B, _ = x.shape
        device  = x.device

        h           = self.input_proj(x)
        h           = self.pos_enc(h)
        causal_mask = self._causal_mask(T, device)

        attn_weights = []
        for layer in self.transformer.layers:
            # Extract attention weights from each layer manually.
            # Pre-norm version.
            h_norm = layer.norm1(h)
            attn_out, attn_w = layer.self_attn(
                h_norm, h_norm, h_norm,
                attn_mask     = causal_mask,
                need_weights  = True,
                average_heads = False,   # return per-head weights
            )
            attn_weights.append(attn_w.detach())   # (B, n_heads, T, T)

            h = h + layer.dropout1(attn_out)
            h = h + layer.dropout2(layer.linear2(
                layer.dropout(layer.activation(layer.linear1(layer.norm2(h))))
            ))

        h      = self.transformer.norm(h)
        output = self.output_proj(h)

        return output, h, attn_weights