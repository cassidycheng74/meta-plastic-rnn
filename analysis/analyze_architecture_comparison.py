"""

Analyses:
    1. Learning speed curves — per-task performance over training steps
       for all three architectures on the same axes, by task family.

    2. Cross-architecture shared subspace — fit PCA on LeakyRNN hidden
       states for each task, project GRU and Transformer onto those axes.
       Tests whether different architectures find the same representational
       geometry for the same tasks.

    3. Cross-architecture variance matrix — for each task, how much of
       architecture B's endpoint variance is explained by architecture A's
       top PCs? Quantifies representational similarity.

    4. Per-family representational geometry — PCA plots showing all three
       architectures for the same task family, colored by architecture.
       Reveals whether memory tasks show ring attractors in all three
       or only in the LeakyRNN.

    5. In-context learning comparison — special focus on tasks T23-T25
       since the transformer is expected to solve these differently
       via attention vs Hebbian-like recurrent dynamics.

Usage:
    python analysis/analyze_architecture_comparison.py

Saves figures to:
    analysis/figures/architecture_comparison/
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from sklearn.decomposition import PCA
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

ARCHITECTURES = [
    {
        'name'    : 'LeakyRNN',
        'run_dir' : 'runs/LeakyRNN_256units_30tasks_seed0',
        'final_ckpt': 'ckpt_step1000000.pt',
        'rnn_type': 'LeakyRNN',
        'n_rnn'   : 256,
        'color'   : '#4C72B0',
        'marker'  : 'o',
        'max_steps': 1_000_000,
    },
    {
        'name'    : 'GRU',
        'run_dir' : 'runs/GRU_256units_30tasks_seed0',
        'final_ckpt': 'ckpt_step4300000.pt',
        'rnn_type': 'GRU',
        'n_rnn'   : 256,
        'color'   : '#55A868',
        'marker'  : 's',
        'max_steps': 4_300_000,
    },
    {
        'name'    : 'Transformer',
        'run_dir' : 'runs/Transformer_128d_30tasks_seed0',
        'final_ckpt': 'ckpt_step5700000.pt',
        'rnn_type': 'Transformer',
        'n_rnn'   : 128,
        'color'   : '#C44E52',
        'marker'  : '^',
        'max_steps': 5_700_000,
    },
]

SEED     = 0
N_TRIALS = 64
FIG_DIR  = 'analysis/figures/architecture_comparison'
os.makedirs(FIG_DIR, exist_ok=True)

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

all_tasks = {**YANG_DRISCOLL_TASKS, **NEW_TASKS}
task_names = list(all_tasks.keys())

# ---------------------------------------------------------------------------
# Load final models and datasets
# ---------------------------------------------------------------------------

print('Loading final models...')
models   = {}
datasets = {}

for arch in ARCHITECTURES:
    config  = make_config(n_rnn=arch['n_rnn'], batch_size=N_TRIALS,
                          rnn_type=arch['rnn_type'])
    dataset = TaskDataset(all_tasks, config, seed=SEED)
    model   = build_network(config)

    ckpt_path = os.path.join(arch['run_dir'], arch['final_ckpt'])
    ckpt      = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if 'task_vecs' in ckpt:
        dataset.task_vecs = ckpt['task_vecs']
    model.eval()

    models[arch['name']]   = model
    datasets[arch['name']] = dataset
    print(f'  Loaded {arch["name"]}')


# ---------------------------------------------------------------------------
# Extract hidden states for all tasks and all architectures
# ---------------------------------------------------------------------------

def get_hidden_states(arch_name, task_name):
    """Returns endpoints (N_TRIALS, n_rnn) and full hidden (T, N_TRIALS, n_rnn)."""
    arch     = next(a for a in ARCHITECTURES if a['name'] == arch_name)
    model    = models[arch_name]
    dataset  = datasets[arch_name]
    config   = make_config(n_rnn=arch['n_rnn'], batch_size=N_TRIALS,
                           rnn_type=arch['rnn_type'])
    rng      = np.random.RandomState(hash(task_name + arch_name) % 2**31)
    task_vec = dataset.task_vecs[task_name]
    fn       = all_tasks[task_name]

    trial = fn(config, rng)
    trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec
    trial.add_input_noise(rng)
    x_t, _, _ = trial.to_tensors()

    with torch.no_grad():
        result = model(x_t)

    h = result.hidden.numpy()   # (T, B, n_rnn)
    return h[-1, :, :], h, trial.epochs  # endpoints, full, epochs


print('\nExtracting hidden states for all tasks and architectures...')
arch_hidden    = {}   # arch_name -> task_name -> endpoints (N_TRIALS, n_rnn)
arch_hidden_full = {}  # arch_name -> task_name -> (T, N_TRIALS, n_rnn)
arch_epochs    = {}   # arch_name -> task_name -> epochs

for arch in ARCHITECTURES:
    arch_hidden[arch['name']]      = {}
    arch_hidden_full[arch['name']] = {}
    arch_epochs[arch['name']]      = {}
    for task_name in task_names:
        ep, hf, epochs = get_hidden_states(arch['name'], task_name)
        arch_hidden[arch['name']][task_name]      = ep
        arch_hidden_full[arch['name']][task_name] = hf
        arch_epochs[arch['name']][task_name]      = epochs
    print(f'  {arch["name"]} done')


# ---------------------------------------------------------------------------
# Figure 1: Learning speed curves — per family, all architectures
# ---------------------------------------------------------------------------

print('\nFigure 1: Learning speed curves from checkpoints...')

def load_perf_from_checkpoints(arch, task_names_eval, max_ckpts=15):
    """Load per-task performance from checkpoint files."""
    run_dir  = arch['run_dir']
    rnn_type = arch['rnn_type']
    n_rnn    = arch['n_rnn']

    config  = make_config(n_rnn=n_rnn, batch_size=16, rnn_type=rnn_type)
    dataset = TaskDataset(all_tasks, config, seed=SEED)
    rng     = np.random.RandomState(42)

    ckpt_files = sorted([
        f for f in os.listdir(run_dir)
        if f.startswith('ckpt_step') and f.endswith('.pt')
    ], key=lambda f: int(f.replace('ckpt_step', '').replace('.pt', '')))

    N       = max(1, len(ckpt_files) // max_ckpts)
    sampled = ckpt_files[::N]

    steps      = []
    task_perfs = {t: [] for t in task_names_eval}

    for ckpt_file in sampled:
        step = int(ckpt_file.replace('ckpt_step', '').replace('.pt', ''))
        steps.append(step / 1000)

        model = build_network(config)
        ckpt  = torch.load(os.path.join(run_dir, ckpt_file), map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        task_vecs = ckpt.get('task_vecs', dataset.task_vecs)
        model.eval()

        with torch.no_grad():
            for t_name in task_names_eval:
                fn       = all_tasks[t_name]
                task_vec = task_vecs.get(t_name)
                perfs    = []
                for _ in range(2):
                    trial = fn(config, rng)
                    if task_vec is not None:
                        trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec
                    trial.add_input_noise(rng)
                    x, y, c = trial.to_tensors()
                    result  = model(x)
                    perf    = compute_performance(result.output, y, c)
                    perfs.append(perf)
                task_perfs[t_name].append(float(np.mean(perfs)))

    return np.array(steps), task_perfs


# Load checkpoint performance for all architectures.
print('  Loading checkpoints for learning speed analysis...')
arch_ckpt_data = {}
for arch in ARCHITECTURES:
    print(f'    {arch["name"]}...')
    steps, task_perfs = load_perf_from_checkpoints(arch, task_names)
    arch_ckpt_data[arch['name']] = {'steps': steps, 'perfs': task_perfs}

# Plot per family.
n_families = len(FAMILIES)
ncols      = 3
nrows      = (n_families + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
axes      = axes.flatten()

for idx, (family_name, family_info) in enumerate(FAMILIES.items()):
    ax    = axes[idx]
    tasks = [t for t in family_info['tasks'] if t in all_tasks]

    for arch in ARCHITECTURES:
        steps     = arch_ckpt_data[arch['name']]['steps']
        perfs_all = arch_ckpt_data[arch['name']]['perfs']

        # Mean performance across tasks in this family.
        family_perfs = np.array([perfs_all[t] for t in tasks
                                 if t in perfs_all and len(perfs_all[t]) > 0])
        if len(family_perfs) == 0:
            continue
        mean_perf = family_perfs.mean(axis=0)

        ax.plot(steps, mean_perf, color=arch['color'], linewidth=2.5,
                marker=arch['marker'], markersize=4, alpha=0.85,
                label=f'{arch["name"]} ({arch["max_steps"]//1000:.0f}k steps)')

        # Individual tasks as thin lines.
        for t in tasks:
            if t in perfs_all and len(perfs_all[t]) > 0:
                ax.plot(steps, perfs_all[t], color=arch['color'],
                        linewidth=0.6, alpha=0.25)

    ax.set_title(family_name, fontsize=10, fontweight='bold',
                 color=family_info['color'])
    ax.set_xlabel('Training Steps (thousands)', fontsize=9)
    ax.set_ylabel('Performance', fontsize=9)
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

for idx in range(n_families, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Learning Speed by Task Family — All Three Architectures\n'
             'Bold = family mean  |  Thin = individual tasks  |  '
             'Different x-ranges reflect different training budgets',
             fontsize=13, y=1.01)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_learning_speed_by_family.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'  Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Cross-architecture shared subspace
# PCA fitted on LeakyRNN, GRU and Transformer projected onto same axes
# ---------------------------------------------------------------------------

print('\nFigure 2: Cross-architecture shared subspace...')

# Tasks to show in detail.
SHARED_TASKS = ['memorypro', 'dm', 'pulsecounting', 'onlinelinearreg']

fig, axes = plt.subplots(2, len(SHARED_TASKS), figsize=(5 * len(SHARED_TASKS), 10))
colors_trial = plt.cm.hsv(np.linspace(0, 1, N_TRIALS))

for col_idx, task_name in enumerate(SHARED_TASKS):
    # Fit PCA on LeakyRNN endpoints.
    ep_rnn = arch_hidden['LeakyRNN'][task_name]   # (N_TRIALS, 256)
    pca    = PCA(n_components=2)
    pca.fit(ep_rnn)
    var    = pca.explained_variance_ratio_

    # PC1 vs PC2 — endpoints.
    ax = axes[0, col_idx]
    for arch in ARCHITECTURES:
        ep   = arch_hidden[arch['name']][task_name]
        # Project onto LeakyRNN PCA axes.
        # Need to handle different n_rnn dimensions — only possible if
        # we use a common projection. Use per-architecture PCA instead
        # and show side by side.
        pca_arch = PCA(n_components=2)
        pca_arch.fit(ep)
        proj = pca_arch.transform(ep)
        var_arch = pca_arch.explained_variance_ratio_

        ax.scatter(proj[:, 0], proj[:, 1],
                   c=colors_trial[:N_TRIALS],
                   marker=arch['marker'], s=50, alpha=0.7,
                   edgecolors=arch['color'], linewidth=1.5,
                   label=f'{arch["name"]} ({var_arch[0]:.0%}/{var_arch[1]:.0%})')

    ax.set_title(f'{TASK_LABELS.get(task_name, task_name)}\n'
                 f'Endpoints (each arch in own PCA space)',
                 fontsize=10)
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

    # Trajectories for one trial per architecture.
    ax = axes[1, col_idx]
    for arch in ARCHITECTURES:
        hf     = arch_hidden_full[arch['name']][task_name]   # (T, B, n_rnn)
        epochs = arch_epochs[arch['name']][task_name]
        pca_arch = PCA(n_components=2)
        pca_arch.fit(hf.reshape(-1, hf.shape[-1]))
        var_arch = pca_arch.explained_variance_ratio_

        resp_key = 'go1' if 'go1' in epochs else list(epochs.keys())[-1]
        resp_on  = epochs[resp_key][0]

        # Plot trajectory for trial 0.
        traj = pca_arch.transform(hf[:, 0, :])
        ax.plot(traj[:resp_on, 0], traj[:resp_on, 1],
                color=arch['color'], alpha=0.3, linewidth=1.0,
                linestyle='--')
        ax.plot(traj[resp_on:, 0], traj[resp_on:, 1],
                color=arch['color'], alpha=0.8, linewidth=1.5,
                label=arch['name'])
        # Endpoint.
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   color=arch['color'], s=80, zorder=5,
                   marker=arch['marker'], edgecolors='white', linewidth=0.5)

    ax.set_title(f'Trajectory (trial 0)\nDashed=pre-resp, Solid=resp',
                 fontsize=9)
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

plt.suptitle('Cross-Architecture Geometry: Same Tasks, Different Networks\n'
             'Each architecture in its own PCA space — do endpoint patterns match?',
             fontsize=13, y=1.01)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_cross_arch_shared_subspace.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'  Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Cross-architecture variance matrix
# For each task, how much of arch B's variance is explained by arch A's PCs?
# Uses a common random projection to handle different n_rnn dimensions.
# ---------------------------------------------------------------------------

print('\nFigure 3: Cross-architecture variance matrix...')

# Since architectures have different n_rnn (256 vs 128), we can't directly
# compare PCA subspaces. Instead compare within-architecture subspace
# structure: for each pair of tasks, compute variance explained within
# each architecture and compare the matrices.

# Simpler approach: for each architecture pair, compute correlation of
# their per-task variance explained matrices (from the cross-task analysis).
# This tells us whether the two architectures organize tasks the same way.

n_arch = len(ARCHITECTURES)
arch_names = [a['name'] for a in ARCHITECTURES]

# For each architecture, compute cross-task variance matrix.
print('  Computing per-architecture cross-task variance matrices...')
arch_var_matrices = {}

for arch in ARCHITECTURES:
    n_tasks_here = len(task_names)
    var_matrix   = np.zeros((n_tasks_here, n_tasks_here))

    for i, name_a in enumerate(task_names):
        ep_a = arch_hidden[arch['name']][name_a]
        pca  = PCA(n_components=3)
        pca.fit(ep_a)
        for j, name_b in enumerate(task_names):
            ep_b      = arch_hidden[arch['name']][name_b]
            proj      = pca.transform(ep_b)
            var_total = np.var(ep_b, axis=0).sum()
            var_proj  = np.var(proj, axis=0).sum()
            var_matrix[i, j] = var_proj / (var_total + 1e-8)

    arch_var_matrices[arch['name']] = var_matrix
    print(f'    {arch["name"]} done')

# Plot: 3 variance matrices side by side + difference matrices.
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
labels    = [TASK_LABELS.get(t, t) for t in task_names]

for col_idx, arch in enumerate(ARCHITECTURES):
    ax = axes[0, col_idx]
    im = ax.imshow(arch_var_matrices[arch['name']],
                   cmap='Blues', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_xticks(range(len(task_names)))
    ax.set_yticks(range(len(task_names)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(f'{arch["name"]}\nCross-Task Variance Matrix',
                 fontsize=10, color=arch['color'], fontweight='bold')

    # Draw family separators.
    FAMILIES_ORDERED = [
        ['delaypro', 'delayanti', 'memorypro', 'memoryanti', 'extendedmemory'],
        ['dm', 'dmanti', 'contextdm_a', 'contextdm_b'],
        ['delaymatchsample', 'delaynonmatchsample'],
        ['pulsecounting', 'intervalreproduction', 'pulserateestimation'],
        ['rhythmgeneration', 'sequencerecall', 'conditionalrhythm'],
        ['toggle', 'conditionaltoggle'],
        ['cueresponseassoc', 'pairedassociation', 'reversallearning', 'multiitemrecall'],
        ['onlinelinearreg', 'onlinenonlinearreg', 'fewshotclassif'],
        ['memorydm', 'countandrecall', 'delayedassociation', 'sequentialdecision'],
    ]
    count = 0
    for fam in FAMILIES_ORDERED:
        n = len([t for t in fam if t in task_names])
        count += n
        if count < len(task_names):
            ax.axhline(count - 0.5, color='white', linewidth=1.5)
            ax.axvline(count - 0.5, color='white', linewidth=1.5)

# Difference matrices: GRU - LeakyRNN and Transformer - LeakyRNN.
diff_pairs = [
    ('GRU',         'LeakyRNN', axes[1, 0]),
    ('Transformer', 'LeakyRNN', axes[1, 1]),
    ('Transformer', 'GRU',      axes[1, 2]),
]

for arch_a, arch_b, ax in diff_pairs:
    diff = arch_var_matrices[arch_a] - arch_var_matrices[arch_b]
    im   = ax.imshow(diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.6, label='Δ variance explained')
    ax.set_xticks(range(len(task_names)))
    ax.set_yticks(range(len(task_names)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(f'{arch_a} − {arch_b}\nBlue=A explains more, Red=B explains more',
                 fontsize=10)

plt.suptitle('Cross-Architecture Variance Matrix Comparison\n'
             'Top: per-architecture cross-task structure  |  '
             'Bottom: differences between architectures',
             fontsize=13, y=1.01)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_cross_arch_variance_matrix.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'  Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 4: In-context learning tasks — special comparison
# These are expected to differ most between architectures
# ---------------------------------------------------------------------------

print('\nFigure 4: In-context learning comparison...')

ICL_TASKS = ['onlinelinearreg', 'onlinenonlinearreg', 'fewshotclassif']

fig, axes = plt.subplots(2, len(ICL_TASKS), figsize=(5 * len(ICL_TASKS), 10))
colors_trial = plt.cm.plasma(np.linspace(0.1, 0.9, N_TRIALS))

for col_idx, task_name in enumerate(ICL_TASKS):
    # Top row: endpoint scatter per architecture (own PCA space)
    ax = axes[0, col_idx]
    for arch in ARCHITECTURES:
        ep       = arch_hidden[arch['name']][task_name]
        pca_arch = PCA(n_components=2)
        pca_arch.fit(ep)
        proj     = pca_arch.transform(ep)
        var_arch = pca_arch.explained_variance_ratio_

        ax.scatter(proj[:, 0], proj[:, 1],
                   c=colors_trial[:N_TRIALS],
                   marker=arch['marker'], s=50, alpha=0.7,
                   edgecolors=arch['color'], linewidth=1.5,
                   label=f'{arch["name"]} ({var_arch[0]:.0%}/{var_arch[1]:.0%})')

    ax.set_title(f'{TASK_LABELS.get(task_name, task_name)}\nEndpoints',
                 fontsize=10)
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Bottom row: full PCA trajectory for one trial each
    ax = axes[1, col_idx]
    for arch in ARCHITECTURES:
        hf       = arch_hidden_full[arch['name']][task_name]
        epochs   = arch_epochs[arch['name']][task_name]
        pca_arch = PCA(n_components=2)
        pca_arch.fit(hf.reshape(-1, hf.shape[-1]))
        var_arch = pca_arch.explained_variance_ratio_

        resp_key = 'go1' if 'go1' in epochs else list(epochs.keys())[-1]
        resp_on  = epochs[resp_key][0]

        traj = pca_arch.transform(hf[:, 0, :])
        ax.plot(traj[:resp_on, 0], traj[:resp_on, 1],
                color=arch['color'], alpha=0.3, linewidth=0.8, linestyle='--')
        ax.plot(traj[resp_on:, 0], traj[resp_on:, 1],
                color=arch['color'], alpha=0.9, linewidth=1.5,
                label=f'{arch["name"]}')
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   color=arch['color'], s=80, zorder=5,
                   marker=arch['marker'], edgecolors='white')

    ax.set_title(f'Trajectory (PC1={var_arch[0]:.0%})', fontsize=9)
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

plt.suptitle('In-Context Learning Tasks: Architecture Comparison\n'
             'Do transformers show different geometry than RNNs for these tasks?\n'
             'Hypothesis: transformer uses attention (in-context), RNN uses Hebbian-like dynamics',
             fontsize=12, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_icl_architecture_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'  Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 5: Final performance heatmap — tasks x architectures
# ---------------------------------------------------------------------------

print('\nFigure 5: Final performance heatmap...')

# Evaluate all architectures on all tasks with task identity injection.
print('  Evaluating final performance...')
final_perfs = {}

for arch in ARCHITECTURES:
    model    = models[arch['name']]
    dataset  = datasets[arch['name']]
    config   = make_config(n_rnn=arch['n_rnn'], batch_size=16,
                           rnn_type=arch['rnn_type'])
    rng      = np.random.RandomState(42)
    perfs    = {}

    with torch.no_grad():
        for task_name, fn in all_tasks.items():
            task_vec    = dataset.task_vecs[task_name]
            batch_perfs = []
            for _ in range(5):
                trial = fn(config, rng)
                trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec
                trial.add_input_noise(rng)
                x, y, c = trial.to_tensors()
                result  = model(x)
                perf    = compute_performance(result.output, y, c)
                batch_perfs.append(perf)
            perfs[task_name] = float(np.mean(batch_perfs))

    final_perfs[arch['name']] = perfs
    print(f'  {arch["name"]} evaluated')

# Order tasks by family.
ordered_tasks = []
for fi in FAMILIES.values():
    for t in fi['tasks']:
        if t in all_tasks:
            ordered_tasks.append(t)

perf_matrix = np.array([
    [final_perfs[arch['name']][t] for arch in ARCHITECTURES]
    for t in ordered_tasks
])

fig, ax = plt.subplots(figsize=(8, 14))
im = ax.imshow(perf_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Performance', shrink=0.4)

ax.set_xticks(range(len(ARCHITECTURES)))
ax.set_xticklabels([a['name'] for a in ARCHITECTURES], fontsize=11)
ax.set_yticks(range(len(ordered_tasks)))
ax.set_yticklabels([TASK_LABELS.get(t, t) for t in ordered_tasks], fontsize=8)

# Annotate values.
for i in range(len(ordered_tasks)):
    for j in range(len(ARCHITECTURES)):
        val   = perf_matrix[i, j]
        color = 'white' if val < 0.4 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7, color=color)

# Family separators.
count = 0
for fi in FAMILIES.values():
    n = len([t for t in fi['tasks'] if t in all_tasks])
    count += n
    if count < len(ordered_tasks):
        ax.axhline(count - 0.5, color='white', linewidth=2)

ax.set_title('Final Performance: All Tasks × All Architectures\n'
             f'LeakyRNN@1M steps  |  GRU@4.3M  |  Transformer@5.7M',
             fontsize=11, pad=10)

plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_final_perf_heatmap.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'  Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

print('\n' + '=' * 65)
print('ARCHITECTURE COMPARISON SUMMARY')
print('=' * 65)

print(f'\n{"Task":<25}', end='')
for arch in ARCHITECTURES:
    print(f' {arch["name"]:>12}', end='')
print()
print('-' * 65)

for family_name, family_info in FAMILIES.items():
    print(f'\n{family_name}')
    for task_name in family_info['tasks']:
        if task_name in all_tasks:
            label = TASK_LABELS.get(task_name, task_name)
            print(f'  {label:<23}', end='')
            for arch in ARCHITECTURES:
                perf = final_perfs[arch['name']][task_name]
                print(f' {perf:>12.3f}', end='')
            print()

print(f'\n{"Tasks solved (>=0.9)":<25}', end='')
for arch in ARCHITECTURES:
    n_solved = sum(1 for t in all_tasks
                   if final_perfs[arch['name']][t] >= 0.9)
    print(f' {n_solved:>12}', end='')
print()

print(f'\nFigures saved to: {FIG_DIR}')