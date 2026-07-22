"""
Design notes:
    - Networks are stateless across trials; hidden state is
      initialized fresh each forward pass.
    - Regularization losses (L1/L2 on activity and weights) are
      computed inside the forward pass and returned alongside
      the output so the training loop can add them to the task loss.
    - The LeakyRNN matches Driscoll et al. 2024 architecture for
      direct comparison: same alpha, softplus activation, diagonal
      weight initialization.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from collections import namedtuple


# ---------------------------------------------------------------------------
# Named output tuple
# ---------------------------------------------------------------------------

NetworkOutput = namedtuple('NetworkOutput', [
    'output',      # (T, B, n_output)  — network predictions
    'hidden',      # (T, B, n_rnn)     — hidden state at every timestep
    'loss_reg',    # scalar            — regularization loss
])


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

def retanh(x: torch.Tensor) -> torch.Tensor:
    """Rectified tanh: max(0, tanh(x))."""
    return torch.clamp(torch.tanh(x), min=0.0)

ACTIVATIONS = {
    'softplus': softplus,
    'tanh'    : torch.tanh,
    'relu'    : F.relu,
    'retanh'  : retanh,
}


# ---------------------------------------------------------------------------
# LeakyRNNCell
# ---------------------------------------------------------------------------

class LeakyRNNCell(nn.Module):
    """
    Single-step leaky RNN cell.

    Dynamics:
        x(t) = (1 - alpha) * x(t-1)
                + alpha * (W_rec @ phi(x(t-1)) + W_in @ u(t) + b)
                + noise

        phi(x) = activation(x)
        output(t) = W_out @ phi(x(t)) + b_out

    x is the pre-activation state (not phi(x)).

    Args:
        n_input:     number of input channels
        n_rnn:       number of recurrent units
        n_output:    number of output channels
        alpha:       dt / tau, leak rate (default 0.2)
        sigma_rec:   std of recurrent noise added each step
        activation:  'softplus', 'tanh', 'relu', or 'retanh'
        w_rec_init:  'diag' (Driscoll default) or 'rand'
        w_rec_coeff: scaling coefficient on recurrent weight init
    """

    def __init__(
        self,
        n_input:    int,
        n_rnn:      int,
        n_output:   int,
        alpha:      float = 0.2,
        sigma_rec:  float = 0.05,
        activation: str   = 'softplus',
        w_rec_init: str   = 'diag',
        w_rec_coeff: float = 1.0,
    ):
        super().__init__()

        self.n_input   = n_input
        self.n_rnn     = n_rnn
        self.n_output  = n_output
        self.alpha     = alpha
        self.sigma_rec = sigma_rec
        self.phi       = ACTIVATIONS[activation]

        # Input weights: (n_rnn, n_input)
        self.W_in  = nn.Linear(n_input, n_rnn, bias=True)

        # Recurrent weights: (n_rnn, n_rnn), no bias (bias is in W_in)
        self.W_rec = nn.Linear(n_rnn, n_rnn, bias=False)

        # Output weights: (n_output, n_rnn)
        self.W_out = nn.Linear(n_rnn, n_output, bias=True)

        self._init_weights(w_rec_init, w_rec_coeff)

    def _init_weights(self, w_rec_init: str, w_rec_coeff: float):
        """Initialize weights to match Driscoll et al."""
        # Input weights: small random
        nn.init.normal_(self.W_in.weight, std=1.0 / np.sqrt(self.n_input))
        nn.init.zeros_(self.W_in.bias)

        # Recurrent weights
        if w_rec_init == 'diag':
            # Diagonal initialization: identity * coeff + small random
            W = w_rec_coeff * torch.eye(self.n_rnn)
            W += 0.1 * torch.randn(self.n_rnn, self.n_rnn) / np.sqrt(self.n_rnn)
            with torch.no_grad():
                self.W_rec.weight.copy_(W)
        elif w_rec_init == 'rand':
            nn.init.normal_(
                self.W_rec.weight,
                std=w_rec_coeff / np.sqrt(self.n_rnn))
        else:
            raise ValueError(f"Unknown w_rec_init: {w_rec_init!r}")

        # Output weights: small random
        nn.init.normal_(self.W_out.weight, std=1.0 / np.sqrt(self.n_rnn))
        nn.init.zeros_(self.W_out.bias)

    def init_hidden(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> torch.Tensor:
        """Return zero initial hidden state (B, n_rnn)."""
        return torch.zeros(batch_size, self.n_rnn, device=device)

    def forward(
        self,
        u:     torch.Tensor,   # (B, n_input)
        x:     torch.Tensor,   # (B, n_rnn) — pre-activation state
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One timestep.

        Returns:
            x_new:  (B, n_rnn)   updated pre-activation state
            out:    (B, n_output) output at this timestep
        """
        phi_x = self.phi(x)

        # Recurrent noise.
        if self.training and self.sigma_rec > 0:
            noise = self.sigma_rec * torch.randn_like(x)
        else:
            noise = 0.0

        x_new = ((1.0 - self.alpha) * x
                 + self.alpha * (self.W_rec(phi_x) + self.W_in(u))
                 + noise)

        out = self.W_out(self.phi(x_new))
        return x_new, out


# ---------------------------------------------------------------------------
# LeakyRNN (full sequence)
# ---------------------------------------------------------------------------

class LeakyRNN(nn.Module):
    """
    Full-sequence leaky RNN.

    Args:
        n_input, n_rnn, n_output, alpha, sigma_rec, activation,
        w_rec_init, w_rec_coeff: passed to LeakyRNNCell

        l1_h:        L1 penalty on hidden activity (encourages sparsity)
        l2_h:        L2 penalty on hidden activity
        l1_weight:   L1 penalty on recurrent weights
        l2_weight:   L2 penalty on recurrent weights
    """

    def __init__(
        self,
        n_input:     int,
        n_rnn:       int,
        n_output:    int,
        alpha:       float = 0.2,
        sigma_rec:   float = 0.05,
        activation:  str   = 'softplus',
        w_rec_init:  str   = 'diag',
        w_rec_coeff: float = 1.0,
        l1_h:        float = 0.0,
        l2_h:        float = 1e-6,
        l1_weight:   float = 0.0,
        l2_weight:   float = 1e-6,
    ):
        super().__init__()

        self.cell = LeakyRNNCell(
            n_input    = n_input,
            n_rnn      = n_rnn,
            n_output   = n_output,
            alpha      = alpha,
            sigma_rec  = sigma_rec,
            activation = activation,
            w_rec_init = w_rec_init,
            w_rec_coeff= w_rec_coeff,
        )

        self.l1_h      = l1_h
        self.l2_h      = l2_h
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(
        self,
        x:       torch.Tensor,            # (T, B, n_input)
        h0:      Optional[torch.Tensor] = None,  # (B, n_rnn)
    ) -> NetworkOutput:
        """
        Run the network over a full trial.

        Args:
            x:   input sequence (T, B, n_input)
            h0:  optional initial hidden state; zeros if None

        Returns:
            NetworkOutput with fields:
                output:   (T, B, n_output)
                hidden:   (T, B, n_rnn)
                loss_reg: scalar tensor
        """
        T, B, _ = x.shape
        device   = x.device

        h = h0 if h0 is not None else self.cell.init_hidden(B, device)

        hiddens = []
        outputs = []

        for t in range(T):
            h, out = self.cell(x[t], h)
            hiddens.append(h)
            outputs.append(out)

        hidden = torch.stack(hiddens, dim=0)   # (T, B, n_rnn)
        output = torch.stack(outputs, dim=0)   # (T, B, n_output)

        loss_reg = self._regularization_loss(hidden)

        return NetworkOutput(output=output, hidden=hidden, loss_reg=loss_reg)

    def _regularization_loss(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute L1/L2 regularization on activity and weights."""
        loss = torch.tensor(0.0, device=hidden.device)

        phi_h = self.cell.phi(hidden)

        if self.l1_h > 0:
            loss = loss + self.l1_h * phi_h.abs().mean()
        if self.l2_h > 0:
            loss = loss + self.l2_h * phi_h.pow(2).mean()

        W_rec = self.cell.W_rec.weight
        if self.l1_weight > 0:
            loss = loss + self.l1_weight * W_rec.abs().mean()
        if self.l2_weight > 0:
            loss = loss + self.l2_weight * W_rec.pow(2).mean()

        return loss

    @property
    def W_rec(self) -> torch.Tensor:
        """Shortcut to recurrent weight matrix (n_rnn, n_rnn)."""
        return self.cell.W_rec.weight

    @property
    def W_in(self) -> torch.Tensor:
        """Shortcut to input weight matrix (n_rnn, n_input)."""
        return self.cell.W_in.weight

    @property
    def W_out(self) -> torch.Tensor:
        """Shortcut to output weight matrix (n_output, n_rnn)."""
        return self.cell.W_out.weight


# ---------------------------------------------------------------------------
# GRUNet (full sequence)
# ---------------------------------------------------------------------------

class GRUNet(nn.Module):
    """
    Uses PyTorch's built-in GRU cell for efficiency, wrapped with the
    same output projection and regularization interface as LeakyRNN.

    Args:
        n_input, n_rnn, n_output: dimensions
        l2_h, l2_weight:          regularization (L1 not typical for GRU)
    """

    def __init__(
        self,
        n_input:   int,
        n_rnn:     int,
        n_output:  int,
        l2_h:      float = 1e-6,
        l2_weight: float = 1e-6,
    ):
        super().__init__()

        self.n_rnn     = n_rnn
        self.l2_h      = l2_h
        self.l2_weight = l2_weight

        self.gru   = nn.GRU(n_input, n_rnn, batch_first=False)
        self.W_out = nn.Linear(n_rnn, n_output, bias=True)

        nn.init.normal_(self.W_out.weight, std=1.0 / np.sqrt(n_rnn))
        nn.init.zeros_(self.W_out.bias)

    def init_hidden(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> torch.Tensor:
        """Return zero initial hidden state (1, B, n_rnn) for nn.GRU."""
        return torch.zeros(1, batch_size, self.n_rnn, device=device)

    def forward(
        self,
        x:  torch.Tensor,                   # (T, B, n_input)
        h0: Optional[torch.Tensor] = None,  # (1, B, n_rnn)
    ) -> NetworkOutput:
        B      = x.shape[1]
        device = x.device

        h0 = h0 if h0 is not None else self.init_hidden(B, device)

        # hidden: (T, B, n_rnn)
        hidden, _ = self.gru(x, h0)
        output    = self.W_out(hidden)   # (T, B, n_output)

        loss_reg = self._regularization_loss(hidden)

        return NetworkOutput(output=output, hidden=hidden, loss_reg=loss_reg)

    def _regularization_loss(self, hidden: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=hidden.device)
        if self.l2_h > 0:
            loss = loss + self.l2_h * hidden.pow(2).mean()
        if self.l2_weight > 0:
            for name, param in self.gru.named_parameters():
                if 'weight' in name:
                    loss = loss + self.l2_weight * param.pow(2).mean()
        return loss


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_network(config: dict) -> nn.Module:
    """
    Config keys used:
        rnn_type:    'LeakyRNN' or 'GRU'
        n_input, n_rnn, n_output
        alpha, sigma_rec, activation
        w_rec_init, w_rec_coeff
        l1_h, l2_h, l1_weight, l2_weight
    """
    rnn_type = config.get('rnn_type', 'LeakyRNN')

    if rnn_type == 'LeakyRNN':
        return LeakyRNN(
            n_input     = config['n_input'],
            n_rnn       = config['n_rnn'],
            n_output    = config['n_output'],
            alpha       = config['alpha'],
            sigma_rec   = config.get('sigma_rec', 0.05),
            activation  = config.get('activation', 'softplus'),
            w_rec_init  = config.get('w_rec_init', 'diag'),
            w_rec_coeff = config.get('w_rec_coeff', 1.0),
            l1_h        = config.get('l1_h', 0.0),
            l2_h        = config.get('l2_h', 1e-6),
            l1_weight   = config.get('l1_weight', 0.0),
            l2_weight   = config.get('l2_weight', 1e-6),
        )
    elif rnn_type == 'GRU':
        return GRUNet(
            n_input    = config['n_input'],
            n_rnn      = config['n_rnn'],
            n_output   = config['n_output'],
            l2_h       = config.get('l2_h', 1e-6),
            l2_weight  = config.get('l2_weight', 1e-6),
        )
    elif rnn_type == 'Transformer':
        from network.transformer import TransformerNet
        return TransformerNet(
            n_input   = config['n_input'],
            n_output  = config['n_output'],
            d_model   = config.get('d_model', 128),
            n_heads   = config.get('n_heads', 4),
            n_layers  = config.get('n_layers', 3),
            d_ff      = config.get('d_ff', 256),
            dropout   = config.get('dropout', 0.1),
            l2_weight = config.get('l2_weight', 1e-6),
        )
    else:
        raise ValueError(f"Unknown rnn_type: {rnn_type!r}. "
                         f"Choose 'LeakyRNN' or 'GRU'.")
