"""

Input channels (11 total):
    0        fixation
    1, 2     angle A  (sin θ, cos θ)
    3, 4     angle B  (sin θ, cos θ)
    5        real A   (scalar)
    6        real B   (scalar)
    7, 8, 9, 10  task identity vector (4-dim ±1, fixed per task per lifetime)

Output channels (5 total):
    0        fixation
    1, 2     angle response (sin φ, cos φ)
    3        real A response (scalar)
    4        real B response (scalar)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, List, Optional, Tuple, Callable


# ---------------------------------------------------------------------------
# Channel index constants — import these in task files
# ---------------------------------------------------------------------------

# Inputs
IN_FIX    = 0
IN_SIN_A  = 1
IN_COS_A  = 2
IN_SIN_B  = 3
IN_COS_B  = 4
IN_REAL_A = 5
IN_REAL_B = 6
IN_CUE_0  = 7
IN_CUE_1  = 8
IN_CUE_2  = 9
IN_CUE_3  = 10

N_INPUT   = 11

# Outputs
OUT_FIX    = 0
OUT_SIN    = 1
OUT_COS    = 2
OUT_REAL_A = 3
OUT_REAL_B = 4

N_OUTPUT   = 5


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------

class Trial:
    """
    A batch of trials for one task.

    Arrays are (T, B, C) — time first, then batch, then channels.

    Args:
        tdim:       number of timesteps
        batch_size: number of trials in the batch
        dt:         timestep duration in ms
        sigma_x:    input noise std (applied by add_input_noise)
        alpha:      dt / tau, used to scale noise correctly
    """

    def __init__(
        self,
        tdim:       int,
        batch_size: int,
        dt:         float = 20.0,
        sigma_x:    float = 0.1,
        alpha:      float = 0.2,
    ):
        self.tdim       = tdim
        self.batch_size = batch_size
        self.dt         = dt

        # Scale noise to account for integration timestep.
        self._sigma_x = sigma_x * np.sqrt(2.0 / alpha)

        # Input array: (T, B, N_INPUT), zeros by default.
        self.x = np.zeros((tdim, batch_size, N_INPUT), dtype=np.float32)

        # Target output: (T, B, N_OUTPUT), zeros by default.
        self.y = np.zeros((tdim, batch_size, N_OUTPUT), dtype=np.float32)

        # Cost mask: (T, B, N_OUTPUT).
        self.c_mask = np.zeros((tdim, batch_size, N_OUTPUT), dtype=np.float32)

        # Epoch dict: maps epoch name -> (start, end) in timesteps.
        self.epochs: Dict[str, Tuple[Optional[int], Optional[int]]] = {}

    # -----------------------------------------------------------------------
    # Convenience setters
    # -----------------------------------------------------------------------

    def set_input(
        self,
        channel:   int,
        value:     float | np.ndarray,
        t_start:   Optional[int] = None,
        t_end:     Optional[int] = None,
        batch_idx: Optional[int | List[int]] = None,
    ):
        """Set input channel to value over [t_start, t_end)."""
        t0 = t_start if t_start is not None else 0
        t1 = t_end   if t_end   is not None else self.tdim
        if batch_idx is None:
            self.x[t0:t1, :, channel] = value
        else:
            self.x[t0:t1, batch_idx, channel] = value

    def set_output(
        self,
        channel:   int,
        value:     float | np.ndarray,
        t_start:   Optional[int] = None,
        t_end:     Optional[int] = None,
        batch_idx: Optional[int | List[int]] = None,
    ):
        """Set output channel to value over [t_start, t_end)."""
        t0 = t_start if t_start is not None else 0
        t1 = t_end   if t_end   is not None else self.tdim
        if batch_idx is None:
            self.y[t0:t1, :, channel] = value
        else:
            self.y[t0:t1, batch_idx, channel] = value

    def set_angle_input(
        self,
        angles:  np.ndarray,
        channel: str,
        t_start: Optional[int] = None,
        t_end:   Optional[int] = None,
    ):
        """
        Encode angles as (sin, cos) into input channels.

        Args:
            angles:  array of shape (batch_size,) in radians
            channel: 'A' or 'B'
        """
        if channel == 'A':
            sin_ch, cos_ch = IN_SIN_A, IN_COS_A
        elif channel == 'B':
            sin_ch, cos_ch = IN_SIN_B, IN_COS_B
        else:
            raise ValueError(f"channel must be 'A' or 'B', got {channel!r}")

        t0 = t_start if t_start is not None else 0
        t1 = t_end   if t_end   is not None else self.tdim
        self.x[t0:t1, :, sin_ch] = np.sin(angles)
        self.x[t0:t1, :, cos_ch] = np.cos(angles)

    def set_angle_output(
        self,
        angles:  np.ndarray,
        t_start: Optional[int] = None,
        t_end:   Optional[int] = None,
    ):
        """Encode angles as (sin, cos) into output channels OUT_SIN, OUT_COS."""
        t0 = t_start if t_start is not None else 0
        t1 = t_end   if t_end   is not None else self.tdim
        self.y[t0:t1, :, OUT_SIN] = np.sin(angles)
        self.y[t0:t1, :, OUT_COS] = np.cos(angles)

    def set_fixation(
        self,
        t_start: Optional[int] = None,
        t_end:   Optional[int] = None,
        value:   float = 1.0,
    ):
        """Set fixation input and output over [t_start, t_end)."""
        t0 = t_start if t_start is not None else 0
        t1 = t_end   if t_end   is not None else self.tdim
        self.x[t0:t1, :, IN_FIX]  = value
        self.y[t0:t1, :, OUT_FIX] = value

    def set_cue_vector(
        self,
        vectors:  np.ndarray,
        t_start:  Optional[int] = None,
        t_end:    Optional[int] = None,
    ):
        """
        Set 4-dim cue vector into inputs 7-10.

        Used for trial-specific cue vectors in associative tasks.

        Args:
            vectors: (batch_size, 4) or (4,) array of ±1 values
        """
        t0 = t_start if t_start is not None else 0
        t1 = t_end   if t_end   is not None else self.tdim
        self.x[t0:t1, :, IN_CUE_0:IN_CUE_3 + 1] = vectors

    # -----------------------------------------------------------------------
    # Cost mask
    # -----------------------------------------------------------------------

    def add_cost_mask(
        self,
        response_on: int,
        pre_weight:  float = 1.0,
        post_weight: float = 5.0,
        pre_on:      int   = 0,
    ):
        """
        Build the cost mask.

        The response epoch (response_on onward) is upweighted relative to
        the pre-response period. Fixation channel gets an extra 2x weight.

        Args:
            response_on:  timestep where response epoch begins
            pre_weight:   cost weight for pre-response period
            post_weight:  cost weight for response period
            pre_on:       ignore the first pre_on timesteps entirely
        """
        self.c_mask[pre_on:response_on, :, :] = pre_weight
        self.c_mask[response_on:,       :, :] = post_weight
        self.c_mask[:, :, OUT_FIX] *= 2.0

    # -----------------------------------------------------------------------
    # Noise
    # -----------------------------------------------------------------------

    def add_input_noise(self, rng: np.random.RandomState):
        """Add Gaussian noise to all input channels."""
        self.x += rng.randn(*self.x.shape).astype(np.float32) * self._sigma_x

    # -----------------------------------------------------------------------
    # Conversion
    # -----------------------------------------------------------------------

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (x, y, c_mask) as float32 PyTorch tensors.

        Shapes:
            x:      (T, B, N_INPUT)
            y:      (T, B, N_OUTPUT)
            c_mask: (T, B, N_OUTPUT)
        """
        x      = torch.from_numpy(self.x)
        y      = torch.from_numpy(self.y)
        c_mask = torch.from_numpy(self.c_mask)
        return x, y, c_mask

    def epoch_slice(self, name: str) -> slice:
        """Return a slice for a named epoch."""
        start, end = self.epochs[name]
        return slice(start, end)


# ---------------------------------------------------------------------------
# Random cue vector helpers
# ---------------------------------------------------------------------------

def random_cue_vector(rng: np.random.RandomState) -> np.ndarray:
    """Return a single random ±1 cue vector of shape (4,)."""
    return rng.choice([-1.0, 1.0], size=4).astype(np.float32)


def random_cue_vectors(
    n:   int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return n random ±1 cue vectors of shape (n, 4)."""
    return rng.choice([-1.0, 1.0], size=(n, 4)).astype(np.float32)


# ---------------------------------------------------------------------------
# TaskDataset
# ---------------------------------------------------------------------------

class TaskDataset(TorchDataset):
    """
    PyTorch Dataset that generates trials on the fly.

    Each call to __getitem__ generates one fresh batch of trials for a
    randomly sampled task, then injects the task identity vector into
    channels 7-10 for the full trial duration.

    Task identity injection:
        Each task is assigned a unique random ±1 vector of shape (4,)
        at initialization. This vector is written into x[:, :, 7:11]
        for every trial of that task, giving the network a consistent
        signal to distinguish tasks with identical sensory inputs
        (e.g. delaypro vs delayanti).

        The vector is fixed within a lifetime but fresh across lifetimes
        (i.e. per TaskDataset instantiation). This matches the task_vec
        convention in the task spec.

    Args:
        task_funcs:        dict mapping task name -> callable -> Trial
        config:            dict of hyperparameters
        batches_per_epoch: artificial epoch length
        task_probs:        optional sampling weights per task
        seed:              random seed for reproducibility
    """

    def __init__(
        self,
        task_funcs:        Dict[str, Callable],
        config:            dict,
        batches_per_epoch: int                       = 1000,
        task_probs:        Optional[Dict[str, float]] = None,
        seed:              Optional[int]              = None,
    ):
        self.task_funcs        = task_funcs
        self.task_names        = list(task_funcs.keys())
        self.config            = config
        self.batches_per_epoch = batches_per_epoch
        self.rng               = np.random.RandomState(seed)

        # Build probability array.
        if task_probs is not None:
            self.probs = np.array(
                [task_probs[n] for n in self.task_names], dtype=np.float64)
            self.probs /= self.probs.sum()
        else:
            n = len(self.task_names)
            self.probs = np.ones(n, dtype=np.float64) / n

        # Generate one unique task identity vector per task.
        # Fixed for this dataset instance (lifetime), fresh on re-instantiation.
        self.task_vecs: Dict[str, np.ndarray] = {
            name: self.rng.choice(
                [-1.0, 1.0], size=4).astype(np.float32)
            for name in self.task_names
        }

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __getitem__(self, idx: int):
        # Sample a task.
        task_name = self.rng.choice(self.task_names, p=self.probs)
        task_fn   = self.task_funcs[task_name]

        # Generate one batch of trials.
        trial = task_fn(self.config, self.rng)

        # Inject task identity vector into channels 7-10 for full trial.
        # This overwrites any zeros already there. For tasks that write
        # trial-specific cues into these channels (T12, T20-T22, T25, T29),
        # those writes happen inside the task function and will overwrite
        # this background during their specific epochs — which is correct.
        task_vec = self.task_vecs[task_name]   # (4,)
        trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec   # broadcast T, B

        # Add input noise after task vec injection so noise is on top.
        trial.add_input_noise(self.rng)

        x, y, c_mask = trial.to_tensors()
        return x, y, c_mask, task_name

    def set_tasks(self, task_names: List[str]):
        """Restrict sampling to a subset of tasks (e.g. for curriculum)."""
        for name in task_names:
            assert name in self.task_funcs, f"Unknown task: {name}"
        self.task_names = task_names
        n = len(task_names)
        self.probs = np.ones(n, dtype=np.float64) / n
        # Note: task_vecs are preserved for all tasks even when curriculum
        # restricts sampling, so resuming full training uses the same vecs.

    def get_task_vec(self, task_name: str) -> np.ndarray:
        """Return the task identity vector for a given task. Shape (4,)."""
        return self.task_vecs[task_name]


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_trials(batch):
    x, y, c_mask, task_name = batch[0]
    return x, y, c_mask, task_name


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Network
    'n_input'      : N_INPUT,
    'n_output'     : N_OUTPUT,
    'n_rnn'        : 256,        # updated default to match PI recommendation

    # Timing
    'dt'           : 20.0,       # ms per timestep
    'tau'          : 100.0,      # membrane time constant in ms
    'alpha'        : 0.2,        # dt / tau

    # Noise
    'sigma_rec'    : 0.05,
    'sigma_x'      : 0.1,

    # Training
    'batch_size'   : 64,
    'learning_rate': 3e-4,       # updated default (was 1e-3)

    # Cost mask weights
    'pre_weight'   : 1.0,
    'post_weight'  : 5.0,

    # Regularization
    'l1_h'         : 0.0,
    'l2_h'         : 1e-6,
    'l1_weight'    : 0.0,
    'l2_weight'    : 1e-6,

    # Architecture
    'activation'   : 'softplus',
    'w_rec_init'   : 'diag',
    'w_rec_coeff'  : 1.0,
}


def make_config(**overrides) -> dict:
    """Return a config dict with defaults overridden by kwargs."""
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(overrides)
    # Recompute alpha if dt or tau changed.
    cfg['alpha'] = cfg['dt'] / cfg['tau']
    return cfg