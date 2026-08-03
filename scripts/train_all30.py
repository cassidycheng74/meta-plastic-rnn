"""
scripts/train_all30.py

Train a LeakyRNN on all 30 tasks (11 Yang/Driscoll + 19 new).
This is the Phase 3 fixed-weight baseline for the meta-plastic-rnn project.

Usage:
    python scripts/train_all30.py                    # default LeakyRNN
    python scripts/train_all30.py --rnn_type GRU     # GRU comparison
    python scripts/train_all30.py --seed 1           # different seed

Output saved to:
    runs/<rnn_type>_<n_rnn>units_<n_tasks>tasks_seed<seed>/
"""

import os
import sys
import argparse
import numpy as np
import torch

from tasks.base import make_config, TaskDataset
from tasks.yang_driscoll import YANG_DRISCOLL_TASKS
from tasks.new_tasks import NEW_TASKS
from network.rnn import build_network
from network.train import Trainer

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Train RNN on all 30 tasks')
parser.add_argument('--n_rnn',       type=int,   default=128,
                    help='Number of RNN units')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to resume from')
parser.add_argument('--seed',        type=int,   default=0,
                    help='Random seed')
parser.add_argument('--max_steps',   type=int,   default=10_000_000,
                    help='Total gradient steps')
parser.add_argument('--batch_size',  type=int,   default=64,
                    help='Batch size per step')
parser.add_argument('--lr',          type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--display_step',type=int,   default=1000,
                    help='Log every N steps')
parser.add_argument('--ckpt_step',   type=int,   default=10_000,
                    help='Save checkpoint every N steps')
parser.add_argument('--target_perf', type=float, default=0.99,
                    help='Early stopping performance threshold')
parser.add_argument('--task_subset', type=str,   default='all30',
                    choices=['all30', 'yang11', 'new19', 'assoc_only',
         'icl_only', 'rhythm_only', 'toggle_only'],
                    help='Which tasks to train on')
parser.add_argument('--save_dir',    type=str,   default=None,
                    help='Override output directory')
parser.add_argument('--rnn_type', type=str, default='LeakyRNN',
                    choices=['LeakyRNN', 'GRU', 'Transformer'])
parser.add_argument('--d_model',  type=int, default=128)
parser.add_argument('--n_heads',  type=int, default=4)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--d_ff',     type=int, default=256)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ---------------------------------------------------------------------------
# Task set
# ---------------------------------------------------------------------------

if args.task_subset == 'all30':
    task_funcs = {**YANG_DRISCOLL_TASKS, **NEW_TASKS}
elif args.task_subset == 'yang11':
    task_funcs = YANG_DRISCOLL_TASKS
elif args.task_subset == 'new19':
    task_funcs = NEW_TASKS
elif args.task_subset == 'rhythm_only':
    task_funcs = {k: NEW_TASKS[k] for k in [
        'rhythmgeneration', 'conditionalrhythm']}
elif args.task_subset == 'toggle_only':
    task_funcs = {k: NEW_TASKS[k] for k in [
        'toggle', 'conditionaltoggle']}

n_tasks = len(task_funcs)
print(f'Task set: {args.task_subset} ({n_tasks} tasks)')
print(f'Tasks: {list(task_funcs.keys())}')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

config = make_config(
    n_rnn        = args.n_rnn,
    batch_size   = args.batch_size,
    rnn_type     = args.rnn_type,
    learning_rate= args.lr,
    sigma_x      = 0.1,
    sigma_rec    = 0.05,
    dt           = 20.0,
    tau          = 100.0,
    # Regularization matching Driscoll defaults.
    l1_h         = 0.0,
    l2_h         = 1e-6,
    l1_weight    = 0.0,
    l2_weight    = 1e-6,
    # Activation and init matching Driscoll.
    activation   = 'softplus',
    w_rec_init   = 'diag',
    w_rec_coeff  = 1.0,
    d_model  = args.d_model,
    n_heads  = args.n_heads,
    n_layers = args.n_layers,
    d_ff     = args.d_ff,
)

# ---------------------------------------------------------------------------
# Save directory
# ---------------------------------------------------------------------------

if args.save_dir is not None:
    save_dir = args.save_dir
else:
    save_dir = os.path.join(
        'runs',
        f'{args.rnn_type}_{args.n_rnn}units_{n_tasks}tasks_seed{args.seed}'
    )

os.makedirs(save_dir, exist_ok=True)
print(f'Saving to: {save_dir}')

# ---------------------------------------------------------------------------
# Dataset, model, trainer
# ---------------------------------------------------------------------------

dataset = TaskDataset(
    task_funcs        = task_funcs,
    config            = config,
    batches_per_epoch = 1000,
    seed              = args.seed,
)

model = build_network(config)
print(f'Model: {args.rnn_type} | '
      f'{sum(p.numel() for p in model.parameters()):,} parameters | '
      f'device: {"cuda" if torch.cuda.is_available() else "cpu"}')

trainer = Trainer(
    model    = model,
    dataset  = dataset,
    config   = config,
    save_dir = save_dir,
)

if args.resume:
    trainer.load_checkpoint(args.resume)
    print(f'Resumed from step {trainer.step}')

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

trainer.train(
    max_steps       = args.max_steps,
    display_step    = args.display_step,
    checkpoint_step = args.ckpt_step,
    target_perf     = args.target_perf,
    eval_tasks      = list(task_funcs.keys()),
)

print(f'\nDone. Results saved to: {save_dir}')