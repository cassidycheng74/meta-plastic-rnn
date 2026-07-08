"""
analysis/plot_training_curves_30tasks.py

Plot per-task performance as a function of training steps
for the 30-task LeakyRNN baseline run.

Usage:
    python analysis/plot_training_curves_30tasks.py

Reads from:
    runs/LeakyRNN_128units_30tasks_seed0_v3/log.json

Saves figures to:
    runs/LeakyRNN_128units_30tasks_seed0_v3/figures/
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUN_DIR  = 'runs/LeakyRNN_128units_30tasks_seed0_v3'
LOG_PATH = os.path.join(RUN_DIR, 'log.json')
FIG_DIR  = os.path.join(RUN_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

# Task families with colors.
FAMILIES = {
    'Yang/Driscoll Memory': {
        'tasks' : ['delaypro', 'delayanti', 'memorypro', 'memoryanti',
                   'extendedmemory'],
        'color' : '#4C72B0',
    },
    'Yang/Driscoll Decision': {
        'tasks' : ['dm', 'dmanti', 'contextdm_a', 'contextdm_b'],
        'color' : '#55A868',
    },
    'Yang/Driscoll Match': {
        'tasks' : ['delaymatchsample', 'delaynonmatchsample'],
        'color' : '#8172B2',
    },
    'Counting/Timing': {
        'tasks' : ['pulsecounting', 'intervalreproduction', 'pulserateestimation'],
        'color' : '#C44E52',
    },
    'Rhythm/Sequence': {
        'tasks' : ['rhythmgeneration', 'sequencerecall', 'conditionalrhythm'],
        'color' : '#DD8452',
    },
    'Bistable': {
        'tasks' : ['toggle', 'conditionaltoggle'],
        'color' : '#937860',
    },
    'Associative': {
        'tasks' : ['cueresponseassoc', 'pairedassociation', 'reversallearning',
                   'multiitemrecall'],
        'color' : '#DA8BC3',
    },
    'In-Context Learning': {
        'tasks' : ['onlinelinearreg', 'onlinenonlinearreg', 'fewshotclassif'],
        'color' : '#8C8C8C',
    },
    'Compositional': {
        'tasks' : ['memorydm', 'countandrecall', 'delayedassociation',
                   'sequentialdecision'],
        'color' : '#64B5CD',
    },
}

TASK_LABELS = {
    'delaypro'           : 'Delay Pro',
    'delayanti'          : 'Delay Anti',
    'memorypro'          : 'Memory Pro',
    'memoryanti'         : 'Memory Anti',
    'extendedmemory'     : 'Ext Memory',
    'dm'                 : 'DM',
    'dmanti'             : 'DM Anti',
    'contextdm_a'        : 'Ctx DM-A',
    'contextdm_b'        : 'Ctx DM-B',
    'delaymatchsample'   : 'DMS',
    'delaynonmatchsample': 'DNMS',
    'pulsecounting'      : 'Pulse Count',
    'intervalreproduction': 'Interval Repro',
    'pulserateestimation': 'Rate Estim',
    'rhythmgeneration'   : 'Rhythm Gen',
    'sequencerecall'     : 'Seq Recall',
    'conditionalrhythm'  : 'Cond Rhythm',
    'toggle'             : 'Toggle',
    'conditionaltoggle'  : 'Cond Toggle',
    'cueresponseassoc'   : 'Cue Assoc',
    'pairedassociation'  : 'Paired Assoc',
    'reversallearning'   : 'Reversal Learn',
    'multiitemrecall'    : 'Multi-Item Recall',
    'onlinelinearreg'    : 'Online Lin Reg',
    'onlinenonlinearreg' : 'Online Nonlin Reg',
    'fewshotclassif'     : 'Few-Shot Classif',
    'memorydm'           : 'Memory DM',
    'countandrecall'     : 'Count & Recall',
    'delayedassociation' : 'Delayed Assoc',
    'sequentialdecision' : 'Sequential Dec',
}

# ---------------------------------------------------------------------------
# Load log
# ---------------------------------------------------------------------------

print(f'Loading log from {LOG_PATH}...')
with open(LOG_PATH) as f:
    log = json.load(f)

steps     = np.array(log['step']) / 1000   # convert to thousands
loss      = np.array(log['loss'])
perf_avg  = np.array(log['perf_avg'])
perf_min  = np.array(log['perf_min'])

# Build per-task performance arrays from the log.
# The log stores perf_avg and perf_min but not per-task.
# We need to reconstruct from the checkpoint evaluation output.
# Since per-task perf isn't stored in log.json, we'll load it
# from the checkpoint files directly.

print(f'Steps logged: {len(steps)}')
print(f'Step range: {steps[0]:.0f}k - {steps[-1]:.0f}k')
print(f'Loss: {loss[0]:.4f} -> {loss[-1]:.4f}')
print(f'Perf avg: {perf_avg[0]:.3f} -> {perf_avg[-1]:.3f}')

# ---------------------------------------------------------------------------
# Figure 1: Overall training summary
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss curve.
ax = axes[0]
ax.semilogy(steps, loss, color='#2c7bb6', linewidth=2)
ax.set_xlabel('Training Steps (thousands)', fontsize=11)
ax.set_ylabel('Loss (log scale)', fontsize=11)
ax.set_title('Training Loss', fontsize=12)
ax.grid(True, alpha=0.3)

# Average performance.
ax = axes[1]
ax.plot(steps, perf_avg, color='#4dac26', linewidth=2, label='avg')
ax.plot(steps, perf_min, color='#d7191c', linewidth=2,
        linestyle='--', label='min')
ax.set_xlabel('Training Steps (thousands)', fontsize=11)
ax.set_ylabel('Performance', fontsize=11)
ax.set_title('Performance (avg and min across tasks)', fontsize=12)
ax.set_ylim([-0.05, 1.05])
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Time per step.
ax = axes[2]
times = np.array(log['time'])
steps_per_sec = np.array(log['step']) / times
ax.plot(steps, steps_per_sec, color='#756bb1', linewidth=2)
ax.set_xlabel('Training Steps (thousands)', fontsize=11)
ax.set_ylabel('Steps per second', fontsize=11)
ax.set_title('Training Speed', fontsize=12)
ax.grid(True, alpha=0.3)

plt.suptitle('30-Task LeakyRNN Training Summary\n'
             f'(128 units, lr=3e-4, {steps[-1]:.0f}k steps)',
             fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_training_summary.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Per-task performance from checkpoints
# ---------------------------------------------------------------------------

print('\nLoading per-task performance from checkpoints...')

import torch
import sys
sys.path.insert(0, '.')

from tasks.base import make_config
from tasks.yang_driscoll import YANG_DRISCOLL_TASKS
from tasks.new_tasks import NEW_TASKS
from network.rnn import build_network
from network.train import compute_performance

config    = make_config(n_rnn=128, batch_size=16)
all_tasks = {**YANG_DRISCOLL_TASKS, **NEW_TASKS}
rng       = __import__('numpy').random.RandomState(42)

# Find available checkpoints.
ckpt_files = sorted([
    f for f in os.listdir(RUN_DIR)
    if f.startswith('ckpt_step') and f.endswith('.pt')
], key=lambda f: int(f.replace('ckpt_step', '').replace('.pt', '')))

print(f'Found {len(ckpt_files)} checkpoints')

# Sample every Nth checkpoint to keep it fast.
N = max(1, len(ckpt_files) // 20)   # at most 20 checkpoints
sampled_ckpts = ckpt_files[::N]

ckpt_steps   = []
task_perfs   = {name: [] for name in all_tasks}

for ckpt_file in sampled_ckpts:
    step = int(ckpt_file.replace('ckpt_step', '').replace('.pt', ''))
    ckpt_steps.append(step / 1000)

    model = build_network(config)
    ckpt  = torch.load(
        os.path.join(RUN_DIR, ckpt_file),
        map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    with torch.no_grad():
        for name, fn in all_tasks.items():
            # Average over 3 batches for stability.
            perfs = []
            for _ in range(3):
                trial = fn(config, rng)
                x, y, c = trial.to_tensors()
                result  = model(x)
                perf    = compute_performance(result.output, y, c,
                                              threshold=0.05)
                perfs.append(perf)
            task_perfs[name].append(float(np.mean(perfs)))

    print(f'  Loaded step {step:,}')

ckpt_steps = np.array(ckpt_steps)

# ---------------------------------------------------------------------------
# Figure 3: Per-family performance curves
# ---------------------------------------------------------------------------

n_families = len(FAMILIES)
ncols      = 3
nrows      = (n_families + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(ncols * 5, nrows * 3.5))
axes = axes.flatten()

for idx, (family_name, family_info) in enumerate(FAMILIES.items()):
    ax     = axes[idx]
    color  = family_info['color']
    tasks  = [t for t in family_info['tasks'] if t in task_perfs]

    for i, task in enumerate(tasks):
        perfs = task_perfs[task]
        alpha = 0.6 + 0.4 * (i / max(1, len(tasks) - 1))
        ax.plot(ckpt_steps, perfs,
                label=TASK_LABELS.get(task, task),
                color=color,
                alpha=alpha,
                linewidth=1.8)

    ax.set_title(family_name, fontsize=10, fontweight='bold', color=color)
    ax.set_xlabel('Steps (thousands)', fontsize=9)
    ax.set_ylabel('Performance', fontsize=9)
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

# Hide unused subplots.
for idx in range(n_families, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Per-Task Performance by Family\n'
             f'30-Task LeakyRNN (128 units, {ckpt_steps[-1]:.0f}k steps)',
             fontsize=13, y=1.01)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_pertask_by_family.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Figure 4: All tasks in one plot, sorted by final performance
# ---------------------------------------------------------------------------

final_perfs = {name: task_perfs[name][-1] for name in all_tasks
               if task_perfs[name]}
sorted_tasks = sorted(final_perfs, key=final_perfs.get, reverse=True)

fig, ax = plt.subplots(figsize=(14, 6))

# Get family color for each task.
task_to_color = {}
for family_info in FAMILIES.values():
    for t in family_info['tasks']:
        task_to_color[t] = family_info['color']

for task in sorted_tasks:
    color = task_to_color.get(task, '#888888')
    ax.plot(ckpt_steps, task_perfs[task],
            color=color, alpha=0.7, linewidth=1.5)

# Annotate final values on right side.
for task in sorted_tasks:
    final = task_perfs[task][-1]
    color = task_to_color.get(task, '#888888')
    ax.annotate(
        f'{TASK_LABELS.get(task, task)} ({final:.2f})',
        xy=(ckpt_steps[-1], final),
        xytext=(ckpt_steps[-1] + 2, final),
        fontsize=6,
        color=color,
        va='center',
    )

ax.set_xlabel('Training Steps (thousands)', fontsize=12)
ax.set_ylabel('Performance', fontsize=12)
ax.set_title('All 30 Tasks: Performance over Training\n'
             '(sorted by final performance, colored by task family)',
             fontsize=12)
ax.set_ylim([-0.05, 1.05])
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3)

# Family legend.
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=info['color'], label=name)
    for name, info in FAMILIES.items()
]
ax.legend(handles=legend_elements, fontsize=8,
          loc='lower left', ncol=2)

plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_all30_performance.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Figure 5: Final performance bar chart sorted by family
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 5))

bar_tasks  = []
bar_perfs  = []
bar_colors = []
bar_labels = []

for family_name, family_info in FAMILIES.items():
    for task in family_info['tasks']:
        if task in final_perfs:
            bar_tasks.append(task)
            bar_perfs.append(final_perfs[task])
            bar_colors.append(family_info['color'])
            bar_labels.append(TASK_LABELS.get(task, task))

x    = np.arange(len(bar_tasks))
bars = ax.bar(x, bar_perfs, color=bar_colors,
              edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Final Performance', fontsize=12)
ax.set_title(f'Final Performance Across All 30 Tasks\n'
             f'(at step {ckpt_steps[-1]:.0f}k, colored by task family)',
             fontsize=12)
ax.set_ylim([0, 1.15])
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

for bar, perf in zip(bars, bar_perfs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{perf:.2f}',
            ha='center', va='bottom', fontsize=7)

# Family legend.
legend_elements = [
    Patch(facecolor=info['color'], label=name)
    for name, info in FAMILIES.items()
]
ax.legend(handles=legend_elements, fontsize=8,
          loc='lower right', ncol=2)

plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_final_performance_bar.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

print('\n' + '=' * 50)
print('FINAL PERFORMANCE SUMMARY')
print('=' * 50)
print(f'{"Task":<25} {"Final":>8} {"Max":>8}')
print('-' * 45)
for task in sorted_tasks:
    perfs  = task_perfs[task]
    final  = perfs[-1]
    maxp   = max(perfs)
    print(f'{TASK_LABELS.get(task, task):<25} {final:>8.3f} {maxp:>8.3f}')

print(f'\nAll figures saved to: {FIG_DIR}')