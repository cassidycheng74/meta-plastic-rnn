"""
Usage:
    python analysis/plot_training_curves_single.py \
        --run_dir runs/GRU_256units_30tasks_seed0 \
        --n_rnn 256 \
        --rnn_type GRU

    python analysis/plot_training_curves_single.py \
        --run_dir runs/Transformer_128d_30tasks_seed0 \
        --n_rnn 128 \
        --rnn_type Transformer

Saves figures to:
    <run_dir>/figures/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch

sys.path.insert(0, '.')

from tasks.base import make_config, TaskDataset, IN_CUE_0, IN_CUE_3
from tasks.yang_driscoll import YANG_DRISCOLL_TASKS
from tasks.new_tasks import NEW_TASKS
from network.rnn import build_network
from network.train import compute_performance

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--run_dir',  type=str, required=True)
parser.add_argument('--n_rnn',    type=int, default=256)
parser.add_argument('--rnn_type', type=str, default='LeakyRNN',
                    choices=['LeakyRNN', 'GRU', 'Transformer'])
parser.add_argument('--seed',     type=int, default=0)
parser.add_argument('--max_ckpts',type=int, default=20)
args = parser.parse_args()

FIG_DIR  = os.path.join(args.run_dir, 'figures')
LOG_PATH = os.path.join(args.run_dir, 'log.json')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

FAMILIES = {
    'Yang/Driscoll Memory': {
        'tasks': ['delaypro', 'delayanti', 'memorypro', 'memoryanti', 'extendedmemory'],
        'color': '#4C72B0',
    },
    'Yang/Driscoll Decision': {
        'tasks': ['dm', 'dmanti', 'contextdm_a', 'contextdm_b'],
        'color': '#55A868',
    },
    'Yang/Driscoll Match': {
        'tasks': ['delaymatchsample', 'delaynonmatchsample'],
        'color': '#8172B2',
    },
    'Counting/Timing': {
        'tasks': ['pulsecounting', 'intervalreproduction', 'pulserateestimation'],
        'color': '#C44E52',
    },
    'Rhythm/Sequence': {
        'tasks': ['rhythmgeneration', 'sequencerecall', 'conditionalrhythm'],
        'color': '#DD8452',
    },
    'Bistable': {
        'tasks': ['toggle', 'conditionaltoggle'],
        'color': '#937860',
    },
    'Associative': {
        'tasks': ['cueresponseassoc', 'pairedassociation', 'reversallearning', 'multiitemrecall'],
        'color': '#DA8BC3',
    },
    'In-Context Learning': {
        'tasks': ['onlinelinearreg', 'onlinenonlinearreg', 'fewshotclassif'],
        'color': '#8C8C8C',
    },
    'Compositional': {
        'tasks': ['memorydm', 'countandrecall', 'delayedassociation', 'sequentialdecision'],
        'color': '#64B5CD',
    },
}

TASK_LABELS = {
    'delaypro': 'Delay Pro', 'delayanti': 'Delay Anti',
    'memorypro': 'Memory Pro', 'memoryanti': 'Memory Anti',
    'extendedmemory': 'Ext Memory', 'dm': 'DM', 'dmanti': 'DM Anti',
    'contextdm_a': 'Ctx DM-A', 'contextdm_b': 'Ctx DM-B',
    'delaymatchsample': 'DMS', 'delaynonmatchsample': 'DNMS',
    'pulsecounting': 'Pulse Count', 'intervalreproduction': 'Interval Repro',
    'pulserateestimation': 'Rate Estim', 'rhythmgeneration': 'Rhythm Gen',
    'sequencerecall': 'Seq Recall', 'conditionalrhythm': 'Cond Rhythm',
    'toggle': 'Toggle', 'conditionaltoggle': 'Cond Toggle',
    'cueresponseassoc': 'Cue Assoc', 'pairedassociation': 'Paired Assoc',
    'reversallearning': 'Reversal Learn', 'multiitemrecall': 'Multi-Item Recall',
    'onlinelinearreg': 'Online Lin Reg', 'onlinenonlinearreg': 'Online Nonlin Reg',
    'fewshotclassif': 'Few-Shot Classif', 'memorydm': 'Memory DM',
    'countandrecall': 'Count & Recall', 'delayedassociation': 'Delayed Assoc',
    'sequentialdecision': 'Sequential Dec',
}

task_to_color = {}
for fi in FAMILIES.values():
    for t in fi['tasks']:
        task_to_color[t] = fi['color']

all_tasks = {**YANG_DRISCOLL_TASKS, **NEW_TASKS}

# ---------------------------------------------------------------------------
# Load log
# ---------------------------------------------------------------------------

print(f'Loading log from {LOG_PATH}...')
with open(LOG_PATH) as f:
    log = json.load(f)

steps    = np.array(log['step']) / 1000
loss     = np.array(log['loss'])
perf_avg = np.array(log['perf_avg'])
perf_min = np.array(log['perf_min'])

print(f'Steps: {steps[0]:.0f}k - {steps[-1]:.0f}k')
print(f'Loss: {loss[0]:.4f} -> {loss[-1]:.4f}')

# ---------------------------------------------------------------------------
# Figure 1: Training summary
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.semilogy(steps, loss, color='#2c7bb6', linewidth=2)
ax.set_xlabel('Training Steps (thousands)', fontsize=11)
ax.set_ylabel('Loss (log scale)', fontsize=11)
ax.set_title('Training Loss', fontsize=12)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(steps, perf_avg, color='#4dac26', linewidth=2, label='avg')
ax.plot(steps, perf_min, color='#d7191c', linewidth=2,
        linestyle='--', label='min')
ax.set_xlabel('Training Steps (thousands)', fontsize=11)
ax.set_ylabel('Performance', fontsize=11)
ax.set_title('Performance (avg and min)', fontsize=12)
ax.set_ylim([-0.05, 1.05])
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle(f'{args.rnn_type} Training Summary\n'
             f'({args.n_rnn} units, {steps[-1]:.0f}k steps)',
             fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_training_summary.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Load checkpoints with task identity injection
# ---------------------------------------------------------------------------

print('\nLoading per-task performance from checkpoints...')

config  = make_config(n_rnn=args.n_rnn, batch_size=16, rnn_type=args.rnn_type)
dataset = TaskDataset(all_tasks, config, seed=args.seed)
rng     = np.random.RandomState(42)

ckpt_files = sorted([
    f for f in os.listdir(args.run_dir)
    if f.startswith('ckpt_step') and f.endswith('.pt')
], key=lambda f: int(f.replace('ckpt_step', '').replace('.pt', '')))

N       = max(1, len(ckpt_files) // args.max_ckpts)
sampled = ckpt_files[::N]
print(f'Found {len(ckpt_files)} checkpoints, sampling {len(sampled)}')

ckpt_steps = []
task_perfs = {name: [] for name in all_tasks}

for ckpt_file in sampled:
    step = int(ckpt_file.replace('ckpt_step', '').replace('.pt', ''))
    ckpt_steps.append(step / 1000)

    model = build_network(config)
    ckpt  = torch.load(os.path.join(args.run_dir, ckpt_file), map_location='cpu')
    model.load_state_dict(ckpt['model_state'])

    task_vecs = ckpt.get('task_vecs', dataset.task_vecs)
    model.eval()

    with torch.no_grad():
        for name, fn in all_tasks.items():
            task_vec = task_vecs.get(name)
            perfs    = []
            for _ in range(3):
                trial = fn(config, rng)
                if task_vec is not None:
                    trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec
                trial.add_input_noise(rng)
                x, y, c = trial.to_tensors()
                result  = model(x)
                perf    = compute_performance(result.output, y, c, threshold=0.05)
                perfs.append(perf)
            task_perfs[name].append(float(np.mean(perfs)))

    print(f'  step {step:,}')

ckpt_steps = np.array(ckpt_steps)

# ---------------------------------------------------------------------------
# Figure 2: Per-family curves
# ---------------------------------------------------------------------------

n_families = len(FAMILIES)
ncols = 3
nrows = (n_families + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
axes = axes.flatten()

for idx, (family_name, family_info) in enumerate(FAMILIES.items()):
    ax    = axes[idx]
    color = family_info['color']
    tasks = [t for t in family_info['tasks'] if t in task_perfs]

    for i, task in enumerate(tasks):
        alpha = 0.6 + 0.4 * (i / max(1, len(tasks) - 1))
        ax.plot(ckpt_steps, task_perfs[task],
                label=TASK_LABELS.get(task, task),
                color=color, alpha=alpha, linewidth=1.8)

    ax.set_title(family_name, fontsize=10, fontweight='bold', color=color)
    ax.set_xlabel('Steps (thousands)', fontsize=9)
    ax.set_ylabel('Performance', fontsize=9)
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

for idx in range(n_families, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Per-Task Performance by Family\n'
             f'{args.rnn_type} ({args.n_rnn} units, {ckpt_steps[-1]:.0f}k steps)',
             fontsize=13, y=1.01)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_pertask_by_family.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Final performance bar chart
# ---------------------------------------------------------------------------

ordered_tasks  = []
ordered_colors = []
for fi in FAMILIES.values():
    for t in fi['tasks']:
        if t in all_tasks:
            ordered_tasks.append(t)
            ordered_colors.append(fi['color'])

final_perfs = {t: task_perfs[t][-1] for t in all_tasks if task_perfs[t]}

fig, ax = plt.subplots(figsize=(14, 5))
x    = np.arange(len(ordered_tasks))
bars = ax.bar(x, [final_perfs[t] for t in ordered_tasks],
              color=ordered_colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([TASK_LABELS.get(t, t) for t in ordered_tasks],
                   rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Final Performance', fontsize=12)
ax.set_title(f'Final Performance: {args.rnn_type}\n'
             f'(at step {ckpt_steps[-1]:.0f}k)', fontsize=12)
ax.set_ylim([0, 1.15])
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

for bar, t in zip(bars, ordered_tasks):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{final_perfs[t]:.2f}',
            ha='center', va='bottom', fontsize=7)

legend_elements = [
    Patch(facecolor=fi['color'], label=name)
    for name, fi in FAMILIES.items()
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right', ncol=2)

plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_final_performance_bar.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

print('\n' + '=' * 50)
print(f'FINAL PERFORMANCE: {args.rnn_type}')
print('=' * 50)
sorted_tasks = sorted(final_perfs, key=final_perfs.get, reverse=True)
for t in sorted_tasks:
    print(f'  {TASK_LABELS.get(t, t):<25} {final_perfs[t]:.3f}')

solved = sum(1 for t in all_tasks if final_perfs.get(t, 0) >= 0.9)
print(f'\nTasks solved (>=0.9): {solved}/30')
print(f'All figures saved to: {FIG_DIR}')