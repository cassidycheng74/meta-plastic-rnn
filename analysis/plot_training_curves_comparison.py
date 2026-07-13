"""
analysis/plot_training_curves_comparison.py

Plot per-task performance curves and a final comparison table
for all three architectures: LeakyRNN, GRU, and Transformer.

Usage:
    python analysis/plot_training_curves_comparison.py

Saves figures to:
    analysis/figures/
"""

import os
import json
import sys
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
# Config
# ---------------------------------------------------------------------------

RUNS = [
    {
        'name'    : 'LeakyRNN',
        'run_dir' : 'runs/LeakyRNN_256units_30tasks_seed0',
        'rnn_type': 'LeakyRNN',
        'n_rnn'   : 256,
        'color'   : '#4C72B0',
        'steps'   : '1M',
    },
    {
        'name'    : 'GRU',
        'run_dir' : 'runs/GRU_256units_30tasks_seed0',
        'rnn_type': 'GRU',
        'n_rnn'   : 256,
        'color'   : '#55A868',
        'steps'   : '4.3M',
    },
    {
        'name'    : 'Transformer',
        'run_dir' : 'runs/Transformer_128d_30tasks_seed0',
        'rnn_type': 'Transformer',
        'n_rnn'   : 128,
        'color'   : '#C44E52',
        'steps'   : '5.7M',
    },
]

SEED    = 0
FIG_DIR = 'analysis/figures'
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
    'delaypro'            : 'Delay Pro',
    'delayanti'           : 'Delay Anti',
    'memorypro'           : 'Memory Pro',
    'memoryanti'          : 'Memory Anti',
    'extendedmemory'      : 'Ext Memory',
    'dm'                  : 'DM',
    'dmanti'              : 'DM Anti',
    'contextdm_a'         : 'Ctx DM-A',
    'contextdm_b'         : 'Ctx DM-B',
    'delaymatchsample'    : 'DMS',
    'delaynonmatchsample' : 'DNMS',
    'pulsecounting'       : 'Pulse Count',
    'intervalreproduction': 'Interval Repro',
    'pulserateestimation' : 'Rate Estim',
    'rhythmgeneration'    : 'Rhythm Gen',
    'sequencerecall'      : 'Seq Recall',
    'conditionalrhythm'   : 'Cond Rhythm',
    'toggle'              : 'Toggle',
    'conditionaltoggle'   : 'Cond Toggle',
    'cueresponseassoc'    : 'Cue Assoc',
    'pairedassociation'   : 'Paired Assoc',
    'reversallearning'    : 'Reversal Learn',
    'multiitemrecall'     : 'Multi-Item Recall',
    'onlinelinearreg'     : 'Online Lin Reg',
    'onlinenonlinearreg'  : 'Online Nonlin Reg',
    'fewshotclassif'      : 'Few-Shot Classif',
    'memorydm'            : 'Memory DM',
    'countandrecall'      : 'Count & Recall',
    'delayedassociation'  : 'Delayed Assoc',
    'sequentialdecision'  : 'Sequential Dec',
}

all_tasks = {**YANG_DRISCOLL_TASKS, **NEW_TASKS}

# ---------------------------------------------------------------------------
# Load per-task performance from checkpoints for each run
# ---------------------------------------------------------------------------

def load_run_perfs(run_info, max_ckpts=20):
    """Load per-task performance from checkpoints with task identity injection."""
    run_dir  = run_info['run_dir']
    rnn_type = run_info['rnn_type']
    n_rnn    = run_info['n_rnn']

    config  = make_config(n_rnn=n_rnn, batch_size=16, rnn_type=rnn_type)
    dataset = TaskDataset(all_tasks, config, seed=SEED)
    rng     = np.random.RandomState(42)

    ckpt_files = sorted([
        f for f in os.listdir(run_dir)
        if f.startswith('ckpt_step') and f.endswith('.pt')
    ], key=lambda f: int(f.replace('ckpt_step', '').replace('.pt', '')))

    N = max(1, len(ckpt_files) // max_ckpts)
    sampled = ckpt_files[::N]

    ckpt_steps = []
    task_perfs = {name: [] for name in all_tasks}

    for ckpt_file in sampled:
        step = int(ckpt_file.replace('ckpt_step', '').replace('.pt', ''))
        ckpt_steps.append(step / 1000)

        model = build_network(config)
        ckpt  = torch.load(os.path.join(run_dir, ckpt_file), map_location='cpu')
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

        print(f'  [{run_info["name"]}] step {step:,}')

    return np.array(ckpt_steps), task_perfs


print('Loading checkpoints for all runs...')
run_data = {}
for run_info in RUNS:
    print(f'\n{run_info["name"]}:')
    steps, perfs = load_run_perfs(run_info)
    run_data[run_info['name']] = {
        'steps' : steps,
        'perfs' : perfs,
        'info'  : run_info,
    }

# ---------------------------------------------------------------------------
# Figure 1: Training loss comparison
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
for run_info in RUNS:
    run_dir = run_info['run_dir']
    with open(os.path.join(run_dir, 'log.json')) as f:
        log = json.load(f)
    steps = np.array(log['step']) / 1000
    loss  = np.array(log['loss'])
    ax.semilogy(steps, loss, color=run_info['color'],
                linewidth=1.5, alpha=0.8, label=run_info['name'])

ax.set_xlabel('Training Steps (thousands)', fontsize=11)
ax.set_ylabel('Loss (log scale)', fontsize=11)
ax.set_title('Training Loss', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
for run_info in RUNS:
    run_dir = run_info['run_dir']
    with open(os.path.join(run_dir, 'log.json')) as f:
        log = json.load(f)
    steps    = np.array(log['step']) / 1000
    perf_avg = np.array(log['perf_avg'])
    ax.plot(steps, perf_avg, color=run_info['color'],
            linewidth=1.5, alpha=0.8,
            label=f'{run_info["name"]} ({run_info["steps"]} steps)')

ax.set_xlabel('Training Steps (thousands)', fontsize=11)
ax.set_ylabel('Perf avg (logged)', fontsize=11)
ax.set_title('Average Performance\n(note: may be inaccurate without task identity in eval)', fontsize=10)
ax.set_ylim([-0.05, 1.05])
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('LeakyRNN vs GRU vs Transformer: Training Dynamics', fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_comparison_training_dynamics.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Final performance comparison bar chart
# ---------------------------------------------------------------------------

# Get final performance for each architecture per task.
final_perfs = {}
for arch_name, data in run_data.items():
    final_perfs[arch_name] = {
        name: data['perfs'][name][-1]
        for name in all_tasks
    }

# Order tasks by family.
ordered_tasks  = []
ordered_colors = []
for family_info in FAMILIES.values():
    for task in family_info['tasks']:
        if task in all_tasks:
            ordered_tasks.append(task)
            ordered_colors.append(family_info['color'])

x      = np.arange(len(ordered_tasks))
width  = 0.25
fig, ax = plt.subplots(figsize=(16, 6))

for i, run_info in enumerate(RUNS):
    arch   = run_info['name']
    perfs  = [final_perfs[arch][t] for t in ordered_tasks]
    offset = (i - 1) * width
    bars   = ax.bar(x + offset, perfs, width,
                    label=f'{arch} ({run_info["steps"]})',
                    color=run_info['color'], alpha=0.8,
                    edgecolor='white', linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels([TASK_LABELS.get(t, t) for t in ordered_tasks],
                   rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Final Performance', fontsize=12)
ax.set_title('Final Performance: LeakyRNN vs GRU vs Transformer\n'
             '(tasks ordered by family, colored by family)', fontsize=12)
ax.set_ylim([0, 1.2])
ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend(fontsize=10, loc='lower right')

# Add family color patches along the top.
family_legend = [
    Patch(facecolor=info['color'], label=name, alpha=0.6)
    for name, info in FAMILIES.items()
]
ax2 = ax.twinx()
ax2.set_yticks([])
ax2.legend(handles=family_legend, fontsize=7, loc='upper left',
           title='Task family', ncol=3, title_fontsize=8)

plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_comparison_final_performance.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Per-family learning curves, one subplot per family,
#           all three architectures overlaid
# ---------------------------------------------------------------------------

n_families = len(FAMILIES)
ncols      = 3
nrows      = (n_families + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
axes = axes.flatten()

for idx, (family_name, family_info) in enumerate(FAMILIES.items()):
    ax    = axes[idx]
    tasks = [t for t in family_info['tasks'] if t in all_tasks]

    for run_info in RUNS:
        arch  = run_info['name']
        data  = run_data[arch]
        steps = data['steps']
        color = run_info['color']

        # Average performance across tasks in this family.
        family_perfs = np.array([
            data['perfs'][t] for t in tasks if t in data['perfs']
        ])
        if len(family_perfs) == 0:
            continue
        mean_perf = family_perfs.mean(axis=0)

        ax.plot(steps, mean_perf, color=color, linewidth=2,
                label=arch, alpha=0.9)

        # Individual tasks as thin lines.
        for t in tasks:
            if t in data['perfs']:
                ax.plot(steps, data['perfs'][t], color=color,
                        linewidth=0.7, alpha=0.3)

    ax.set_title(family_name, fontsize=10, fontweight='bold',
                 color=family_info['color'])
    ax.set_xlabel('Steps (thousands)', fontsize=9)
    ax.set_ylabel('Performance', fontsize=9)
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

for idx in range(n_families, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Per-Family Performance: LeakyRNN vs GRU vs Transformer\n'
             '(bold = family mean, thin = individual tasks)',
             fontsize=13, y=1.01)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_comparison_by_family.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Figure 4: Heatmap of final performance — tasks x architectures
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 12))

arch_names = [r['name'] for r in RUNS]
perf_matrix = np.array([
    [final_perfs[arch][t] for arch in arch_names]
    for t in ordered_tasks
])

im = ax.imshow(perf_matrix, aspect='auto', cmap='RdYlGn',
               vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Performance', shrink=0.5)

ax.set_xticks(range(len(arch_names)))
ax.set_xticklabels(arch_names, fontsize=11)
ax.set_yticks(range(len(ordered_tasks)))
ax.set_yticklabels([TASK_LABELS.get(t, t) for t in ordered_tasks], fontsize=8)

# Add text annotations.
for i in range(len(ordered_tasks)):
    for j in range(len(arch_names)):
        val = perf_matrix[i, j]
        color = 'white' if val < 0.4 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7, color=color)

# Draw family separators.
task_count = 0
for family_info in FAMILIES.values():
    n = len([t for t in family_info['tasks'] if t in all_tasks])
    task_count += n
    if task_count < len(ordered_tasks):
        ax.axhline(task_count - 0.5, color='white', linewidth=2)

ax.set_title('Final Performance Heatmap\nLeakyRNN vs GRU vs Transformer',
             fontsize=12, pad=10)

plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_comparison_heatmap.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

print('\n' + '=' * 65)
print('FINAL PERFORMANCE COMPARISON')
print('=' * 65)
print(f'{"Task":<25} {"LeakyRNN":>10} {"GRU":>10} {"Transformer":>12}')
print('-' * 65)

for family_name, family_info in FAMILIES.items():
    print(f'\n{family_name}')
    for task in family_info['tasks']:
        if task in all_tasks:
            rnn  = final_perfs['LeakyRNN'][task]
            gru  = final_perfs['GRU'][task]
            trf  = final_perfs['Transformer'][task]
            label = TASK_LABELS.get(task, task)
            print(f'  {label:<23} {rnn:>10.3f} {gru:>10.3f} {trf:>12.3f}')

print('\n' + '-' * 65)
solved = {arch: sum(1 for t in all_tasks if final_perfs[arch][t] >= 0.9)
          for arch in ['LeakyRNN', 'GRU', 'Transformer']}
print(f'  {"Tasks solved (>=0.9)":<23} '
      f'{solved["LeakyRNN"]:>10} {solved["GRU"]:>10} {solved["Transformer"]:>12}')

print(f'\nAll figures saved to: {FIG_DIR}')