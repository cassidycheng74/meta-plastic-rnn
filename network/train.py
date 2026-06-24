"""
Training loop for meta-plastic-rnn.

Handles:
    - Single-task and multi-task training
    - Logging of loss, regularization, and per-task performance
    - Checkpointing
    - Early stopping on target performance
    - Curriculum learning via staged task introduction

Usage:
    from network.train import Trainer
    from network.rnn import build_network
    from tasks.base import make_config, TaskDataset
    from tasks.yang_driscoll import TASK_FUNCS

    config  = make_config(n_rnn=128, rnn_type='LeakyRNN')
    model   = build_network(config)
    dataset = TaskDataset(TASK_FUNCS, config)
    trainer = Trainer(model, dataset, config, save_dir='runs/exp_01')
    trainer.train(max_steps=1_000_000)
"""

from __future__ import annotations

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
from collections import defaultdict

from tasks.base import collate_trials, N_OUTPUT


# ---------------------------------------------------------------------------
# Performance metric
# ---------------------------------------------------------------------------

def compute_performance(
    output:   torch.Tensor,   # (T, B, n_output)
    target:   torch.Tensor,   # (T, B, n_output)
    c_mask:   torch.Tensor,   # (T, B, n_output)
    threshold: float = 0.2,
) -> float:
    """
    Fraction of trials where the mean squared error in the response
    epoch falls below threshold.

    Only timesteps with c_mask > 1 (i.e. the response epoch) are
    evaluated, matching Driscoll's get_perf() logic.

    Returns a float in [0, 1].
    """
    # Response epoch: where c_mask > 1 on any output channel.
    resp_mask = (c_mask > 1.0).any(dim=-1)   # (T, B)

    if not resp_mask.any():
        return 0.0

    # MSE per trial over response timesteps.
    sq_err = ((output - target) ** 2).mean(dim=-1)   # (T, B)
    # Mean over response timesteps per trial.
    resp_mse = (sq_err * resp_mask.float()).sum(dim=0) / (
        resp_mask.float().sum(dim=0).clamp(min=1))     # (B,)

    perf = (resp_mse < threshold).float().mean().item()
    return perf


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def masked_mse_loss(
    output:  torch.Tensor,   # (T, B, n_output)
    target:  torch.Tensor,   # (T, B, n_output)
    c_mask:  torch.Tensor,   # (T, B, n_output)
) -> torch.Tensor:
    """
    Weighted mean squared error loss.

    Each element is weighted by its c_mask value, so the response
    epoch contributes more to the total loss than the fixation period.
    """
    sq_err     = (output - target) ** 2          # (T, B, n_output)
    weighted   = sq_err * c_mask                 # (T, B, n_output)
    # Normalize by total mask weight so loss scale is comparable
    # across tasks with different trial lengths.
    loss = weighted.sum() / c_mask.sum().clamp(min=1.0)
    return loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training loop.

    Args:
        model:      nn.Module with forward(x) -> NetworkOutput
        dataset:    TaskDataset instance
        config:     hyperparameter dict (from make_config())
        save_dir:   directory for checkpoints and logs
        device:     'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(
        self,
        model:    nn.Module,
        dataset,
        config:   dict,
        save_dir: str  = 'runs/default',
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

        # Scheduler: reduce LR if performance plateaus.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode    = 'max',
            factor  = 0.5,
            patience= 20,
            verbose = True,
        )

        # Logging.
        self.log = defaultdict(list)
        self.step = 0

        # DataLoader — batch_size=1 because TaskDataset already
        # returns a full trial batch per item.
        self.loader = DataLoader(
            dataset,
            batch_size  = 1,
            shuffle     = False,
            collate_fn  = collate_trials,
            num_workers = 0,
        )

        os.makedirs(save_dir, exist_ok=True)
        self._save_config()

    # -----------------------------------------------------------------------
    # Main training entry point
    # -----------------------------------------------------------------------

    def train(
        self,
        max_steps:       int   = 1_000_000,
        display_step:    int   = 1_000,
        checkpoint_step: int   = 10_000,
        target_perf:     float = 0.95,
        eval_tasks:      Optional[List[str]] = None,
    ):
        """
        Train the model.

        Args:
            max_steps:       total gradient steps
            display_step:    log and print every N steps
            checkpoint_step: save checkpoint every N steps
            target_perf:     stop early if min per-task perf exceeds this
            eval_tasks:      tasks to evaluate at each display step
                             (None = all tasks in dataset)
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: "
              f"{sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Max steps: {max_steps:,}  |  "
              f"Display every: {display_step:,}")
        print("-" * 60)

        t_start = time.time()
        loader_iter = iter(self.loader)

        while self.step < max_steps:
            # Refresh iterator when exhausted.
            try:
                x, y, c_mask, task_name = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                x, y, c_mask, task_name = next(loader_iter)

            x      = x.to(self.device)
            y      = y.to(self.device)
            c_mask = c_mask.to(self.device)

            # Forward + backward.
            loss, loss_reg = self._train_step(x, y, c_mask)

            self.step += 1

            # Logging and display.
            if self.step % display_step == 0:
                elapsed = time.time() - t_start
                perfs   = self._evaluate(eval_tasks)
                perf_min = min(perfs.values()) if perfs else 0.0
                perf_avg = np.mean(list(perfs.values())) if perfs else 0.0

                self.log['step'].append(self.step)
                self.log['loss'].append(loss)
                self.log['loss_reg'].append(loss_reg)
                self.log['perf_min'].append(perf_min)
                self.log['perf_avg'].append(perf_avg)
                self.log['time'].append(elapsed)

                self._print_status(elapsed, loss, loss_reg, perfs)
                self.scheduler.step(perf_avg)

                # Early stopping.
                if perf_min > target_perf:
                    print(f"\nTarget performance {target_perf:.2f} reached "
                          f"at step {self.step:,}. Stopping.")
                    break

            # Checkpointing.
            if self.step % checkpoint_step == 0:
                self.save_checkpoint(f'ckpt_step{self.step}.pt')

        # Final save.
        self.save_checkpoint('ckpt_final.pt')
        self._save_log()
        print("\nTraining complete.")

    # -----------------------------------------------------------------------
    # Single gradient step
    # -----------------------------------------------------------------------

    def _train_step(
        self,
        x:      torch.Tensor,
        y:      torch.Tensor,
        c_mask: torch.Tensor,
    ):
        self.model.train()
        self.optimizer.zero_grad()

        result   = self.model(x)
        task_loss = masked_mse_loss(result.output, y, c_mask)
        total_loss = task_loss + result.loss_reg

        total_loss.backward()

        # Gradient clipping — same as Driscoll.
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return task_loss.item(), result.loss_reg.item()

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def _evaluate(
        self,
        task_names: Optional[List[str]] = None,
        n_batches:  int = 4,
    ) -> Dict[str, float]:
        """
        Evaluate performance on each task.

        Runs n_batches trials per task (in test mode) and averages perf.
        Returns dict of task_name -> performance.
        """
        self.model.eval()
        perfs = {}

        names = task_names or list(self.dataset.task_funcs.keys())

        with torch.no_grad():
            for name in names:
                task_fn = self.dataset.task_funcs[name]
                batch_perfs = []

                for _ in range(n_batches):
                    trial = task_fn(
                        self.dataset.config,
                        self.dataset.rng,
                    )
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
        """
        Restrict training to a subset of tasks.
        Call this to implement staged curriculum learning.

        Example:
            # Start with easy tasks
            trainer.set_curriculum(['fdgo', 'fdanti'])
            trainer.train(max_steps=100_000)

            # Add harder tasks
            trainer.set_curriculum(['fdgo', 'fdanti', 'delaygo', 'delayanti'])
            trainer.train(max_steps=200_000)
        """
        self.dataset.set_tasks(task_names)
        print(f"Curriculum set to: {task_names}")

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'step'        : self.step,
            'model_state' : self.model.state_dict(),
            'optim_state' : self.optimizer.state_dict(),
            'log'         : dict(self.log),
            'config'      : self.config,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])
        self.step = ckpt['step']
        self.log  = defaultdict(list, ckpt['log'])
        print(f"Loaded checkpoint from {path} (step {self.step:,})")

    # -----------------------------------------------------------------------
    # Logging helpers
    # -----------------------------------------------------------------------

    def _print_status(
        self,
        elapsed:  float,
        loss:     float,
        loss_reg: float,
        perfs:    Dict[str, float],
    ):
        perf_min = min(perfs.values()) if perfs else 0.0
        perf_avg = np.mean(list(perfs.values())) if perfs else 0.0

        print(f"\nStep {self.step:>8,}  |  "
              f"Time {elapsed:6.0f}s  |  "
              f"Loss {loss:.4f}  |  "
              f"Reg {loss_reg:.4f}  |  "
              f"Perf avg {perf_avg:.2f}  min {perf_min:.2f}")

        # Per-task performance.
        for name, perf in sorted(perfs.items()):
            print(f"  {name:<25} perf {perf:.2f}")

    def _save_config(self):
        path = os.path.join(self.save_dir, 'config.json')
        # Convert non-serializable values to strings.
        safe = {k: (v if isinstance(v, (int, float, str, bool, list))
                    else str(v))
                for k, v in self.config.items()}
        with open(path, 'w') as f:
            json.dump(safe, f, indent=2)

    def _save_log(self):
        path = os.path.join(self.save_dir, 'log.json')
        with open(path, 'w') as f:
            json.dump(dict(self.log), f, indent=2)