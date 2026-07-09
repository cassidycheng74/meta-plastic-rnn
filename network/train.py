"""
network/train.py

Training loop for meta-plastic-rnn.

Handles:
    - Single-task and multi-task training
    - Logging of loss, regularization, and per-task performance
    - Checkpointing with periodic log saving including task_vecs
    - Early stopping on target performance
    - Curriculum learning via staged task introduction

Key fixes in this version:
    - _evaluate() now injects task identity vectors so evaluation matches training
    - save_checkpoint() saves task_vecs so they can be restored on resume
    - load_checkpoint() restores task_vecs to dataset
    - Per-task performance logged at every display step
    - Unbuffered print output (use python3 -u for real-time log streaming)
"""

from __future__ import annotations

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from collections import defaultdict

from tasks.base import collate_trials, N_OUTPUT, IN_CUE_0, IN_CUE_3


# ---------------------------------------------------------------------------
# Performance metric
# ---------------------------------------------------------------------------

def compute_performance(
    output:    torch.Tensor,   # (T, B, n_output)
    target:    torch.Tensor,   # (T, B, n_output)
    c_mask:    torch.Tensor,   # (T, B, n_output)
    threshold: float = 0.05,
) -> float:
    """
    Fraction of trials where the mean squared error in the response
    epoch falls below threshold.

    Only timesteps with c_mask > 1 (the response epoch) are evaluated.
    Returns a float in [0, 1].
    """
    resp_mask = (c_mask > 1.0).any(dim=-1)   # (T, B)

    if not resp_mask.any():
        return 0.0

    sq_err   = ((output - target) ** 2).mean(dim=-1)   # (T, B)
    resp_mse = (sq_err * resp_mask.float()).sum(dim=0) / (
        resp_mask.float().sum(dim=0).clamp(min=1))      # (B,)

    perf = (resp_mse < threshold).float().mean().item()
    return perf


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def masked_mse_loss(
    output:  torch.Tensor,
    target:  torch.Tensor,
    c_mask:  torch.Tensor,
) -> torch.Tensor:
    """
    Weighted MSE loss. Response epoch upweighted by c_mask.
    Normalized by total mask weight for comparable scale across tasks.
    """
    sq_err   = (output - target) ** 2
    weighted = sq_err * c_mask
    loss     = weighted.sum() / c_mask.sum().clamp(min=1.0)
    return loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training loop.

    Args:
        model:    nn.Module with forward(x) -> NetworkOutput
        dataset:  TaskDataset instance
        config:   hyperparameter dict (from make_config())
        save_dir: directory for checkpoints and logs
        device:   'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(
        self,
        model:    nn.Module,
        dataset,
        config:   dict,
        save_dir: str           = 'runs/default',
        device:   Optional[str] = None,
    ):
        self.model    = model
        self.dataset  = dataset
        self.config   = config
        self.save_dir = save_dir

        # Device.
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Optimizer.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
        )

        # Scheduler: reduce LR on plateau.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode    = 'max',
            factor  = 0.5,
            patience= 20,
        )

        # Logging.
        self.log  = defaultdict(list)
        self.step = 0

        # DataLoader.
        self.loader = DataLoader(
            dataset,
            batch_size  = 1,
            shuffle     = False,
            collate_fn  = collate_trials,
            num_workers = 0,
        )

        # Small config for fast evaluation.
        self._eval_config = {**config, 'batch_size': 16}

        os.makedirs(save_dir, exist_ok=True)
        self._save_config()

    # -----------------------------------------------------------------------
    # Main training entry point
    # -----------------------------------------------------------------------

    def train(
        self,
        max_steps:       int   = 1_000_000,
        display_step:    int   = 5_000,
        checkpoint_step: int   = 50_000,
        target_perf:     float = 1.1,
        eval_tasks:      Optional[List[str]] = None,
    ):
        """
        Train the model.

        Args:
            max_steps:       total gradient steps
            display_step:    log and print every N steps
            checkpoint_step: save checkpoint every N steps
            target_perf:     stop early if min per-task perf exceeds this
                             (set to 1.1 to disable early stopping)
            eval_tasks:      tasks to evaluate at each display step
        """
        print(f"Training on {self.device}", flush=True)
        print(f"Model parameters: "
              f"{sum(p.numel() for p in self.model.parameters()):,}", flush=True)
        print(f"Max steps: {max_steps:,}  |  "
              f"Display every: {display_step:,}  |  "
              f"Checkpoint every: {checkpoint_step:,}", flush=True)
        print("-" * 60, flush=True)

        t_start     = time.time()
        loader_iter = iter(self.loader)

        while self.step < max_steps:
            try:
                x, y, c_mask, task_name = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                x, y, c_mask, task_name = next(loader_iter)

            x      = x.to(self.device)
            y      = y.to(self.device)
            c_mask = c_mask.to(self.device)

            loss, loss_reg = self._train_step(x, y, c_mask)
            self.step += 1

            # Logging and display.
            if self.step % display_step == 0:
                elapsed  = time.time() - t_start
                perfs    = self._evaluate(eval_tasks)
                perf_min = min(perfs.values()) if perfs else 0.0
                perf_avg = float(np.mean(list(perfs.values()))) if perfs else 0.0

                self.log['step'].append(self.step)
                self.log['loss'].append(loss)
                self.log['loss_reg'].append(loss_reg)
                self.log['perf_min'].append(perf_min)
                self.log['perf_avg'].append(perf_avg)
                self.log['time'].append(elapsed)

                # Log per-task performance so analysis doesn't need checkpoints.
                for name, perf in perfs.items():
                    self.log[f'perf_{name}'].append(perf)

                self._print_status(elapsed, loss, loss_reg, perfs)
                self.scheduler.step(perf_avg)

                # Early stopping.
                if perf_min > target_perf:
                    print(f"\nTarget performance {target_perf:.2f} reached "
                          f"at step {self.step:,}. Stopping.", flush=True)
                    break

            # Checkpointing.
            if self.step % checkpoint_step == 0:
                self.save_checkpoint(f'ckpt_step{self.step}.pt')

        # Final save.
        self.save_checkpoint('ckpt_final.pt')
        self._save_log()
        print("\nTraining complete.", flush=True)

    # -----------------------------------------------------------------------
    # Single gradient step
    # -----------------------------------------------------------------------

    def _train_step(self, x, y, c_mask):
        self.model.train()
        self.optimizer.zero_grad()

        result     = self.model(x)
        task_loss  = masked_mse_loss(result.output, y, c_mask)
        total_loss = task_loss + result.loss_reg

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return task_loss.item(), result.loss_reg.item()

    # -----------------------------------------------------------------------
    # Evaluation — injects task identity vectors to match training
    # -----------------------------------------------------------------------

    def _evaluate(
        self,
        task_names: Optional[List[str]] = None,
        n_batches:  int = 2,
    ) -> Dict[str, float]:
        """
        Evaluate performance on each task.

        Injects the task identity vector into channels 7-10 for each trial,
        matching what TaskDataset does during training. Without this, tasks
        that depend on the identity signal (e.g. delayanti) will appear to
        have zero performance during evaluation even when they are learned.
        """
        self.model.eval()
        perfs = {}
        names = task_names or list(self.dataset.task_funcs.keys())

        with torch.no_grad():
            for name in names:
                task_fn  = self.dataset.task_funcs[name]
                task_vec = self.dataset.task_vecs.get(name)
                batch_perfs = []

                for _ in range(n_batches):
                    trial = task_fn(self._eval_config, self.dataset.rng)

                    # Inject task identity vector — critical for correct eval.
                    if task_vec is not None:
                        trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec

                    trial.add_input_noise(self.dataset.rng)
                    x, y, c_mask = trial.to_tensors()

                    x      = x.to(self.device)
                    y      = y.to(self.device)
                    c_mask = c_mask.to(self.device)

                    result = self.model(x)
                    perf   = compute_performance(result.output, y, c_mask)
                    batch_perfs.append(perf)

                perfs[name] = float(np.mean(batch_perfs))

        return perfs

    # -----------------------------------------------------------------------
    # Curriculum learning
    # -----------------------------------------------------------------------

    def set_curriculum(self, task_names: List[str]):
        """Restrict training to a subset of tasks."""
        self.dataset.set_tasks(task_names)
        print(f"Curriculum set to {len(task_names)} tasks: {task_names}",
              flush=True)

    # -----------------------------------------------------------------------
    # Checkpointing — saves task_vecs so evaluation stays consistent on resume
    # -----------------------------------------------------------------------

    def save_checkpoint(self, filename: str):
        """
        Save model, optimizer, log, config, and task identity vectors.

        task_vecs are saved so that on resume, evaluation uses the same
        identity vectors as training — otherwise resumed evaluation would
        use freshly generated vectors that don't match the trained model.
        """
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'step'       : self.step,
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'log'        : dict(self.log),
            'config'     : self.config,
            'task_vecs'  : self.dataset.task_vecs,
        }, path)
        self._save_log()
        print(f"  Checkpoint saved: {path}", flush=True)

    def load_checkpoint(self, path: str):
        """
        Load model weights, optimizer state, log, and task identity vectors.

        Restores task_vecs to the dataset so evaluation uses the same
        identity vectors the model was trained with.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])
        self.step = ckpt['step']
        self.log  = defaultdict(list, ckpt['log'])

        # Restore task identity vectors if present.
        if 'task_vecs' in ckpt:
            self.dataset.task_vecs = ckpt['task_vecs']
            print(f"  Restored task_vecs for "
                  f"{len(ckpt['task_vecs'])} tasks", flush=True)
        else:
            print("  Warning: checkpoint has no task_vecs — "
                  "evaluation may be inaccurate for identity-dependent tasks",
                  flush=True)

        print(f"Loaded checkpoint from {path} (step {self.step:,})", flush=True)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _print_status(self, elapsed, loss, loss_reg, perfs):
        perf_min = min(perfs.values()) if perfs else 0.0
        perf_avg = float(np.mean(list(perfs.values()))) if perfs else 0.0

        print(f"\nStep {self.step:>8,}  |  "
              f"Time {elapsed:6.0f}s  |  "
              f"Loss {loss:.4f}  |  "
              f"Reg {loss_reg:.4f}  |  "
              f"Perf avg {perf_avg:.2f}  min {perf_min:.2f}",
              flush=True)

        for name, perf in sorted(perfs.items()):
            print(f"  {name:<25} perf {perf:.2f}", flush=True)

    def _save_config(self):
        path = os.path.join(self.save_dir, 'config.json')
        safe = {
            k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
            for k, v in self.config.items()
        }
        with open(path, 'w') as f:
            json.dump(safe, f, indent=2)

    def _save_log(self):
        path = os.path.join(self.save_dir, 'log.json')
        with open(path, 'w') as f:
            json.dump(dict(self.log), f, indent=2)