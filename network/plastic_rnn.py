"""
Main idea:
    Each synapse has a fixed weight W and a plastic component alpha * A,
    where A is a Hebbian trace that updates within/across trials and
    alpha is a learned per-synapse plasticity coefficient.

    r[t]   = phi(h[t])                          post-activation rates
    eff_W  = W_rec + alpha * A[t-1]             effective weight
    h[t]   = (1-tau_decay)*h[t-1]
             + tau_decay*(eff_W @ r[t-1] + W_in @ x[t] + b)
    A[t]   = clip((1-eta)*A[t-1] + eta * outer(r[t], r[t-1].detach()))
    out[t] = W_out @ r[t] + b_out

Across-trial persistence:
    A is carried across trials within a lifetime. Between lifetimes it
    resets to zero. This allows the network to accumulate synaptic changes
    across many trials — enabling across-trial associative learning.

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from collections import namedtuple

from network.rnn import NetworkOutput, ACTIVATIONS, softplus


# ---------------------------------------------------------------------------
# Named tuple for plastic network output
# ---------------------------------------------------------------------------

PlasticOutput = namedtuple('PlasticOutput', [
    'output',      # (T, B, n_output)     network predictions
    'hidden',      # (T, B, n_rnn)        hidden states
    'hebb',        # (B, n_rnn, n_rnn)    Hebbian trace at end of sequence
    'loss_reg',    # scalar               regularization loss
])


# ---------------------------------------------------------------------------
# PlasticRNN
# ---------------------------------------------------------------------------

class PlasticRNN(nn.Module):
    """
    Leaky RNN with Hebbian plasticity (Miconi et al. 2018).

    Args:
        n_input:      input dimension (11 in unified format)
        n_rnn:        number of recurrent units
        n_output:     output dimension (5)
        alpha_init:   initial value for plasticity coefficients
                      (scalar, all synapses start equal, near zero)
        eta_init:     initial Hebbian trace decay rate in (0, 1)
        learn_eta:    if True, eta is a learned parameter
        hebb_clip:    clip Hebbian trace values to [-hebb_clip, hebb_clip]
        activation:   activation function name
        sigma_rec:    recurrent noise std (training only)
        l2_h:         L2 regularization on hidden activity
        l2_weight:    L2 regularization on weights
        l2_alpha:     L2 regularization on plasticity coefficients
        w_rec_init:   recurrent weight initialization ('diag' or 'random')
        w_rec_coeff:  scale factor for initial recurrent weights
        dt:           timestep duration in ms
        tau:          membrane time constant in ms
    """

    def __init__(
        self,
        n_input:     int,
        n_rnn:       int,
        n_output:    int,
        alpha_init:  float = 0.0,
        eta_init:    float = 0.01,
        learn_eta:   bool  = True,
        hebb_clip:   float = 0.5,
        activation:  str   = 'softplus',
        sigma_rec:   float = 0.05,
        l2_h:        float = 1e-6,
        l2_weight:   float = 1e-6,
        l2_alpha:    float = 1e-4,
        w_rec_init:  str   = 'diag',
        w_rec_coeff: float = 0.5,
        dt:          float = 20.0,
        tau:         float = 100.0,
    ):
        super().__init__()

        self.n_input    = n_input
        self.n_rnn      = n_rnn
        self.n_output   = n_output
        self.hebb_clip  = hebb_clip
        self.sigma_rec  = sigma_rec
        self.l2_h       = l2_h
        self.l2_weight  = l2_weight
        self.l2_alpha   = l2_alpha
        self.act        = ACTIVATIONS[activation]
        self.tau_decay  = dt / tau      # fraction of new input per step

        # ---- Fixed recurrent weights ----
        W_rec = self._init_w_rec(n_rnn, w_rec_init, w_rec_coeff)
        self.W_rec = nn.Parameter(W_rec)

        # ---- Input and output projections ----
        self.W_in  = nn.Linear(n_input, n_rnn,    bias=True)
        self.W_out = nn.Linear(n_rnn,   n_output, bias=True)

        # ---- Plasticity coefficients ----
        # alpha: (n_rnn, n_rnn) one per synapse.
        # Initialized near zero so network starts close to fixed-weight baseline.
        self.alpha_raw = nn.Parameter(
            torch.full((n_rnn, n_rnn), float(alpha_init)))

        # ---- Hebbian decay rate ----
        if learn_eta:
            # Parameterize via sigmoid so eta stays in (0, 1).
            eta_raw = float(np.log(eta_init / (1.0 - eta_init + 1e-8)))
            self.eta_raw   = nn.Parameter(torch.tensor(eta_raw))
            self.learn_eta = True
        else:
            self.register_buffer('eta_fixed', torch.tensor(float(eta_init)))
            self.learn_eta = False

        self._init_weights()

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def _init_w_rec(self, n: int, init: str, coeff: float) -> torch.Tensor:
        if init == 'diag':
            W = torch.zeros(n, n)
            W.fill_diagonal_(coeff)
        else:
            W = torch.randn(n, n) * coeff / np.sqrt(n)
        return W

    def _init_weights(self):
        nn.init.normal_(self.W_in.weight,  std=1.0 / np.sqrt(self.n_input))
        nn.init.zeros_(self.W_in.bias)
        nn.init.normal_(self.W_out.weight, std=1.0 / np.sqrt(self.n_rnn))
        nn.init.zeros_(self.W_out.bias)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def eta(self) -> torch.Tensor:
        """Effective Hebbian decay rate in (0, 1)."""
        if self.learn_eta:
            return torch.sigmoid(self.eta_raw)
        return self.eta_fixed

    @property
    def alpha(self) -> torch.Tensor:
        """Plasticity coefficients (n_rnn, n_rnn). Can be + or -."""
        return self.alpha_raw

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -----------------------------------------------------------------------
    # State initialization
    # -----------------------------------------------------------------------

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero hidden state. Shape: (B, n_rnn)."""
        return torch.zeros(batch_size, self.n_rnn, device=device)

    def init_hebb(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Zero Hebbian trace. Shape: (B, n_rnn, n_rnn)."""
        return torch.zeros(batch_size, self.n_rnn, self.n_rnn, device=device)

    # -----------------------------------------------------------------------
    # Single timestep
    # -----------------------------------------------------------------------

    def step(
        self,
        x:     torch.Tensor,    # (B, n_input)
        h:     torch.Tensor,    # (B, n_rnn)
        A:     torch.Tensor,    # (B, n_rnn, n_rnn)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One timestep of the plastic RNN.

        Returns:
            h_new: (B, n_rnn) updated hidden state
            A_new: (B, n_rnn, n_rnn) updated Hebbian trace
            r_new: (B, n_rnn) post-activation rates
        """
        r_prev = self.act(h)   # (B, n_rnn) pre-update rates

        # Effective recurrent weight: W_rec + alpha * A[t-1]
        # W_rec:  (n_rnn, n_rnn)       -> unsqueeze to (1, n_rnn, n_rnn)
        # alpha:  (n_rnn, n_rnn)       -> unsqueeze to (1, n_rnn, n_rnn)
        # A:      (B, n_rnn, n_rnn)
        # eff_W:  (B, n_rnn, n_rnn)
        eff_W = self.W_rec.unsqueeze(0) + self.alpha.unsqueeze(0) * A

        # Recurrent drive: eff_W @ r_prev
        # r_prev: (B, n_rnn) -> (B, n_rnn, 1)
        # output: (B, n_rnn, 1) -> (B, n_rnn)
        rec_input = torch.bmm(eff_W, r_prev.unsqueeze(-1)).squeeze(-1)

        # Input drive.
        inp_input = self.W_in(x)   # (B, n_rnn)

        # Recurrent noise (training only).
        noise = torch.zeros_like(h)
        if self.training and self.sigma_rec > 0:
            noise = torch.randn_like(h) * self.sigma_rec

        # Leaky integration.
        h_new = ((1.0 - self.tau_decay) * h
                 + self.tau_decay * (rec_input + inp_input)
                 + noise)

        r_new = self.act(h_new)   # (B, n_rnn)

        # Hebbian update: A[t] = (1-eta)*A[t-1] + eta * outer(r[t], r[t-1])
        # Detach r_prev in the outer product to avoid retaining graph
        # across the A -> h -> A chain within a trial.
        eta   = self.eta
        outer = torch.bmm(
            r_new.unsqueeze(-1),
            r_prev.detach().unsqueeze(1))   # (B, n_rnn, n_rnn)
        A_new = (1.0 - eta) * A + eta * outer

        # Clip to prevent unbounded growth.
        A_new = torch.clamp(A_new, -self.hebb_clip, self.hebb_clip)

        return h_new, A_new, r_new

    # -----------------------------------------------------------------------
    # Full sequence forward pass
    # -----------------------------------------------------------------------

    def forward(self, x, h0=None, A0=None):
        T, B, _ = x.shape
        device  = x.device

        h = h0 if h0 is not None else self.init_hidden(B, device)
        A = A0 if A0 is not None else self.init_hebb(B, device)

        hiddens = []
        outputs = []

        for t in range(T):
            h, A, r = self.step(x[t], h, A)
            out = self.W_out(r)
            hiddens.append(h)
            outputs.append(out)
            # Detach A every 50 timesteps to prevent graph buildup
            # within a single trial. Loses some gradient signal but
            # dramatically reduces memory.
            if t % 50 == 49:
                A = A.detach()

        hidden_seq = torch.stack(hiddens, dim=0)
        output_seq = torch.stack(outputs, dim=0)
        loss_reg   = self._regularization_loss(hidden_seq)

        return PlasticOutput(output=output_seq, hidden=hidden_seq,
                            hebb=A, loss_reg=loss_reg)

    # -----------------------------------------------------------------------
    # Regularization
    # -----------------------------------------------------------------------

    def _regularization_loss(self, hidden_seq: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=hidden_seq.device)

        if self.l2_h > 0:
            loss = loss + self.l2_h * hidden_seq.pow(2).mean()

        if self.l2_weight > 0:
            loss = (loss
                    + self.l2_weight * self.W_rec.pow(2).mean()
                    + self.l2_weight * self.W_in.weight.pow(2).mean()
                    + self.l2_weight * self.W_out.weight.pow(2).mean())

        # Penalize large plasticity coefficients — keeps alpha interpretable
        # and prevents the plastic term from dominating the fixed weights.
        if self.l2_alpha > 0:
            loss = loss + self.l2_alpha * self.alpha.pow(2).mean()

        return loss


# ---------------------------------------------------------------------------
# LifetimeState — manages h and A across trials
# ---------------------------------------------------------------------------

class LifetimeState:
    """
    Manages hidden state and Hebbian trace across trials within a lifetime.

    Details:
        - h is always detached when carried across trials
        - A is detached every truncate_every trials. Between truncations,
          gradients flow back through A across multiple trials, allowing
          the outer loop to learn how the Hebbian trace should evolve.
        - At the start of a new lifetime, both h and A reset to zero.

    Usage:
        state = LifetimeState(model, batch_size=32, device=device)

        for trial_idx in range(n_trials):
            result = model(x, h0=state.h, A0=state.A)
            loss   = masked_mse_loss(result.output, y, c) + result.loss_reg
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            state.update(result, trial_idx)

        state.reset()  # start of next lifetime
    """

    def __init__(
        self,
        model:          PlasticRNN,
        batch_size:     int,
        device:         torch.device,
        truncate_every: int = 5,
    ):
        self.model          = model
        self.batch_size     = batch_size
        self.device         = device
        self.truncate_every = truncate_every
        self.trial_count    = 0
        self.reset()

    def reset(self):
        """Reset state at the start of a new lifetime."""
        self.h = self.model.init_hidden(self.batch_size, self.device)
        self.A = self.model.init_hebb(self.batch_size, self.device)
        self.trial_count = 0

    def update(self, result: PlasticOutput, trial_idx: int):
        # h always detached.
        self.h = result.hidden[-1].detach()
        self.A = result.hebb.detach()
        self.trial_count += 1


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_plastic_network(config: dict) -> PlasticRNN:
    """
    Relevant config keys beyond standard make_config:
        alpha_init:   initial plasticity coefficient (default 0.0)
        eta_init:     Hebbian decay rate (default 0.01)
        learn_eta:    whether eta is learned (default True)
        hebb_clip:    Hebbian trace clip value (default 0.5)
        l2_alpha:     L2 on plasticity coefficients (default 1e-4)
        w_rec_coeff:  recurrent weight scale (default 0.5 for plastic net)
    """
    return PlasticRNN(
        n_input     = config['n_input'],
        n_rnn       = config['n_rnn'],
        n_output    = config['n_output'],
        alpha_init  = config.get('alpha_init',  0.0),
        eta_init    = config.get('eta_init',    0.01),
        learn_eta   = config.get('learn_eta',   True),
        hebb_clip   = config.get('hebb_clip',   0.5),
        activation  = config.get('activation',  'softplus'),
        sigma_rec   = config.get('sigma_rec',   0.05),
        l2_h        = config.get('l2_h',        1e-6),
        l2_weight   = config.get('l2_weight',   1e-6),
        l2_alpha    = config.get('l2_alpha',    1e-4),
        w_rec_init  = config.get('w_rec_init',  'diag'),
        w_rec_coeff = config.get('w_rec_coeff', 0.5),
        dt          = config.get('dt',          20.0),
        tau         = config.get('tau',         100.0),
    )