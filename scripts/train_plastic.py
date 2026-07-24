"""
scripts/train_plastic.py

Training structure:
    Each "lifetime" = N trials of a randomly sampled task.
    The Hebbian trace A persists across trials within a lifetime,
    accumulating synaptic changes that enable within-lifetime learning.
    At the end of each lifetime, A resets to zero.

    The outer loop (gradient descent) learns:
        W_rec, W_in, W_out  -- fixed weights
        alpha               -- per-synapse plasticity coefficients
        eta                 -- Hebbian decay rate

    The inner loop (Hebbian updates) learns within each lifetime:
        A                   -- fast weight matrix (resets each lifetime)

Key design:
    h and A are always detached between trials after backward() is called. 
    The plasticity signal comes from backprop through the within-trial Hebbian updates (A[t] depends on
    r[t] and r[t-1] within the same trial forward pass).

Logging:
    Two performance metrics are tracked at each display step:
    - perf_avg_fixed:   performance with A=0 (tests fixed weights only)
    - perf_avg_plastic: performance after N trials of Hebbian learning
    Delta (plastic - fixed) shows how much the Hebbian trace helps.

Usage:
    python scripts/train_plastic.py
    python scripts/train_plastic.py --n_rnn 256 --n_trials 20
    python scripts/train_plastic.py --task_subset icl_only
    python scripts/train_plastic.py --resume runs/PlasticRNN_.../ckpt_lifetime10000.pt
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

sys.path.insert(0, '.')

from tasks.base import make_config, TaskDataset, IN_CUE_0, IN_CUE_3
from tasks.yang_driscoll import YANG_DRISCOLL_TASKS
from tasks.new_tasks import NEW_TASKS
from network.plastic_rnn import PlasticRNN, LifetimeState, build_plastic_network
from network.train import masked_mse_loss, compute_performance

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Train PlasticRNN')
parser.add_argument('--n_rnn',         type=int,   default=256)
parser.add_argument('--seed',          type=int,   default=0)
parser.add_argument('--n_lifetimes',   type=int,   default=100_000)
parser.add_argument('--n_trials',      type=int,   default=20,
                    help='Trials per lifetime')
parser.add_argument('--batch_size',    type=int,   default=32)
parser.add_argument('--lr',            type=float, default=3e-4)
parser.add_argument('--alpha_init',    type=float, default=0.0)
parser.add_argument('--eta_init',      type=float, default=0.01)
parser.add_argument('--l2_alpha',      type=float, default=1e-4)
parser.add_argument('--hebb_clip',     type=float, default=0.5)
parser.add_argument('--w_rec_coeff',   type=float, default=0.5)
parser.add_argument('--display_every', type=int,   default=500)
parser.add_argument('--ckpt_every',    type=int,   default=5_000)
parser.add_argument('--task_subset', type=str, default='all30',
                    choices=['all30', 'yang11', 'new19', 'assoc_only',
                             'icl_only', 'rhythm_only', 'toggle_only'])
parser.add_argument('--save_dir',      type=str,   default=None)
parser.add_argument('--resume',        type=str,   default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Task set
# ---------------------------------------------------------------------------

if args.task_subset == 'all30':
    task_funcs = {**YANG_DRISCOLL_TASKS, **NEW_TASKS}
elif args.task_subset == 'yang11':
    task_funcs = YANG_DRISCOLL_TASKS
elif args.task_subset == 'new19':
    task_funcs = NEW_TASKS
elif args.task_subset == 'assoc_only':
    task_funcs = {k: NEW_TASKS[k] for k in [
        'cueresponseassoc', 'pairedassociation',
        'reversallearning', 'multiitemrecall']}
elif args.task_subset == 'icl_only':
    task_funcs = {k: NEW_TASKS[k] for k in [
        'onlinelinearreg', 'onlinenonlinearreg', 'fewshotclassif']}
elif args.task_subset == 'rhythm_only':
    task_funcs = {k: NEW_TASKS[k] for k in [
        'rhythmgeneration', 'conditionalrhythm',
        'toggle', 'conditionaltoggle']}
elif args.task_subset == 'toggle_only':
    task_funcs = {k: NEW_TASKS[k] for k in [
        'toggle', 'conditionaltoggle']}

n_tasks = len(task_funcs)
print(f'Task set: {args.task_subset} ({n_tasks} tasks)', flush=True)
print(f'Tasks: {list(task_funcs.keys())}', flush=True)

# ---------------------------------------------------------------------------
# Config and save dir
# ---------------------------------------------------------------------------

config = make_config(
    n_rnn        = args.n_rnn,
    batch_size   = args.batch_size,
    sigma_rec    = 0.05,
    sigma_x      = 0.1,
    dt           = 20.0,
    tau          = 100.0,
    activation   = 'softplus',
    w_rec_init   = 'diag',
    w_rec_coeff  = args.w_rec_coeff,
    l2_h         = 1e-6,
    l2_weight    = 1e-6,
    l2_alpha     = args.l2_alpha,
    alpha_init   = args.alpha_init,
    eta_init     = args.eta_init,
    learn_eta    = True,
    hebb_clip    = args.hebb_clip,
    learning_rate= args.lr,
)

if args.save_dir is None:
    args.save_dir = os.path.join(
        'runs',
        f'PlasticRNN_{args.n_rnn}units_{n_tasks}tasks_'
        f'T{args.n_trials}_seed{args.seed}'
    )
os.makedirs(args.save_dir, exist_ok=True)
print(f'Saving to: {args.save_dir}', flush=True)

with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
    safe = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in config.items()}
    safe.update({
        'n_trials_per_lifetime': args.n_trials,
        'task_subset'          : args.task_subset,
    })
    json.dump(safe, f, indent=2)

# ---------------------------------------------------------------------------
# Dataset and model
# ---------------------------------------------------------------------------

dataset = TaskDataset(task_funcs, config, seed=args.seed)

model = build_plastic_network(config).to(device)

n_fixed   = sum(p.numel() for n, p in model.named_parameters()
                if 'alpha' not in n and 'eta' not in n)
n_plastic = model.alpha.numel() + (1 if model.learn_eta else 0)

print(f'Model: PlasticRNN | {model.n_params:,} total params', flush=True)
print(f'  Fixed weights: {n_fixed:,}', flush=True)
print(f'  Plastic params (alpha + eta): {n_plastic:,}', flush=True)
print(f'  Alpha shape: {model.alpha.shape}', flush=True)
print(f'  Eta learned: {model.learn_eta}', flush=True)
print(f'  Device: {device}', flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
log       = defaultdict(list)
lifetime  = 0

# ---------------------------------------------------------------------------
# Evaluation helpers (defined before training loop)
# ---------------------------------------------------------------------------

def evaluate_fixed(model, dataset, task_funcs, device, config, n_batches=3):
    """
    eval with zero Hebbian trace — tests fixed weights only.
    """
    model.eval()
    perfs = {}
    rng   = np.random.RandomState(99)
    eval_config = {**config, 'batch_size': 16}

    with torch.no_grad():
        for name, fn in task_funcs.items():
            task_vec    = dataset.task_vecs[name]
            batch_perfs = []
            for _ in range(n_batches):
                trial = fn(eval_config, rng)
                trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec
                trial.add_input_noise(rng)
                x, y, c = trial.to_tensors()
                # A0=None -> zeros, no Hebbian contribution.
                result  = model(x.to(device))
                perf    = compute_performance(
                    result.output, y.to(device), c.to(device))
                batch_perfs.append(perf)
            perfs[name] = float(np.mean(batch_perfs))

    model.train()
    return perfs


def evaluate_plastic(model, dataset, task_funcs, device, config,
                     n_trials=None, n_batches=3):
    """
    eval with Hebbian learning across n_trials.
    """
    if n_trials is None:
        n_trials = args.n_trials

    model.eval()
    perfs = {}
    rng   = np.random.RandomState(42)
    eval_config = {**config, 'batch_size': 16}

    with torch.no_grad():
        for name, fn in task_funcs.items():
            task_vec    = dataset.task_vecs[name]
            batch_perfs = []
            for _ in range(n_batches):
                h = model.init_hidden(16, device)
                A = model.init_hebb(16, device)
                # Accumulate Hebbian trace over n_trials.
                for t_idx in range(n_trials):
                    trial = fn(eval_config, rng)
                    trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec
                    trial.add_input_noise(rng)
                    x, y, c = trial.to_tensors()
                    result  = model(x.to(device), h0=h, A0=A)
                    h = result.hidden[-1].detach()
                    A = result.hebb.detach()
                # Evaluate on the last trial.
                perf = compute_performance(
                    result.output, y.to(device), c.to(device))
                batch_perfs.append(perf)
            perfs[name] = float(np.mean(batch_perfs))

    model.train()
    return perfs


def save_log(log, save_dir):
    with open(os.path.join(save_dir, 'log.json'), 'w') as f:
        json.dump(dict(log), f, indent=2)


def save_checkpoint(model, optimizer, lifetime, log, task_vecs,
                    save_dir, filename):
    path = os.path.join(save_dir, filename)
    torch.save({
        'lifetime'   : lifetime,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'log'        : dict(log),
        'task_vecs'  : task_vecs,
    }, path)
    save_log(log, save_dir)
    print(f'  Checkpoint saved: {path}', flush=True)


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------

if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    lifetime = ckpt['lifetime']
    log      = defaultdict(list, ckpt['log'])
    if 'task_vecs' in ckpt:
        dataset.task_vecs = ckpt['task_vecs']
    print(f'Resumed from {args.resume} (lifetime {lifetime:,})', flush=True)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print(f'\nTraining on {device}', flush=True)
print(f'n_lifetimes={args.n_lifetimes:,}  |  '
      f'n_trials={args.n_trials}  |  '
      f'batch_size={args.batch_size}', flush=True)
print('-' * 60, flush=True)

t_start = time.time()

while lifetime < args.n_lifetimes:

    # Sample one task for this entire lifetime.
    task_name = dataset.rng.choice(dataset.task_names, p=dataset.probs)
    task_fn   = task_funcs[task_name]
    task_vec  = dataset.task_vecs[task_name]

    # Initialize lifetime state (h=0, A=0).
    state = LifetimeState(
        model,
        batch_size = args.batch_size,
        device     = device,
    )

    lifetime_loss = 0.0
    lifetime_perf = 0.0

    # ---- Inner loop: N trials per lifetime ----
    for trial_idx in range(args.n_trials):

        trial_config = {**config, 'batch_size': args.batch_size}
        rng = np.random.RandomState(
            int(lifetime * args.n_trials + trial_idx) % 2**31)

        trial = task_fn(trial_config, rng)
        trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec
        trial.add_input_noise(rng)
        x_t, y_t, c_t = trial.to_tensors()
        x_t = x_t.to(device)
        y_t = y_t.to(device)
        c_t = c_t.to(device)

        # Forward pass with current Hebbian state.
        result = model(x_t, h0=state.h, A0=state.A)

        # Loss and gradient step.
        task_loss  = masked_mse_loss(result.output, y_t, c_t)
        total_loss = task_loss + result.loss_reg
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Accumulate metrics.
        lifetime_loss += task_loss.item()
        with torch.no_grad():
            perf = compute_performance(result.output, y_t, c_t)
            lifetime_perf += perf

        # Update state — always detaches after backward.
        state.update(result, trial_idx)

    # ---- End of lifetime ----
    lifetime_loss /= args.n_trials
    lifetime_perf /= args.n_trials
    lifetime      += 1

    # ---- Logging and display ----
    if lifetime % args.display_every == 0:
        elapsed = time.time() - t_start

        fixed_perfs   = evaluate_fixed(model, dataset, task_funcs,
                                       device, config)
        plastic_perfs = evaluate_plastic(model, dataset, task_funcs,
                                         device, config)

        perf_avg_fixed   = float(np.mean(list(fixed_perfs.values())))
        perf_avg_plastic = float(np.mean(list(plastic_perfs.values())))

        log['lifetime'].append(lifetime)
        log['loss'].append(lifetime_loss)
        log['perf_avg_fixed'].append(perf_avg_fixed)
        log['perf_avg_plastic'].append(perf_avg_plastic)
        log['time'].append(elapsed)
        log['eta'].append(model.eta.item())
        log['alpha_mean'].append(model.alpha.abs().mean().item())
        log['alpha_max'].append(model.alpha.abs().max().item())
        log['last_task'].append(task_name)

        for name, perf in plastic_perfs.items():
            log[f'plastic_{name}'].append(perf)
        for name, perf in fixed_perfs.items():
            log[f'fixed_{name}'].append(perf)

        print(f'\nLifetime {lifetime:>8,}  |  '
              f'Time {elapsed:6.0f}s  |  '
              f'Loss {lifetime_loss:.4f}  |  '
              f'Fixed {perf_avg_fixed:.2f}  '
              f'Plastic {perf_avg_plastic:.2f}  '
              f'Delta {perf_avg_plastic - perf_avg_fixed:+.2f}',
              flush=True)
        print(f'  eta={model.eta.item():.4f}  '
              f'|alpha|_mean={model.alpha.abs().mean().item():.5f}  '
              f'|alpha|_max={model.alpha.abs().max().item():.4f}',
              flush=True)

        # Show per-task breakdown — highlight large deltas.
        for name in sorted(plastic_perfs):
            fp    = fixed_perfs.get(name, 0.0)
            pp    = plastic_perfs[name]
            delta = pp - fp
            marker = ' +++' if delta > 0.1 else (' ---' if delta < -0.1 else '')
            print(f'  {name:<25} fixed={fp:.2f}  '
                  f'plastic={pp:.2f}  delta={delta:+.2f}{marker}',
                  flush=True)

        save_log(log, args.save_dir)

    if lifetime % args.ckpt_every == 0:
        save_checkpoint(model, optimizer, lifetime, log,
                        dataset.task_vecs, args.save_dir,
                        f'ckpt_lifetime{lifetime}.pt')

# Final save.
save_checkpoint(model, optimizer, lifetime, log,
                dataset.task_vecs, args.save_dir, 'ckpt_final.pt')
print('\nTraining complete.', flush=True)