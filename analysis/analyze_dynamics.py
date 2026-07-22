"""
Analyses:
    1. PCA trajectories per task
    2. Pro vs anti shared subspace (ring attractor test)
    3. Cross-task variance matrix (30x30)
    4. Unit variance matrix
    5. Global PCA across all tasks
    6. Compositional task subspace

Usage:
    python analysis/analyze_dynamics.py

Saves figures to:
    runs/LeakyRNN_256units_30tasks_seed0/figures/dynamics/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
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

RUN_DIR  = 'runs/LeakyRNN_256units_30tasks_seed0'
CKPT     = 'ckpt_step1000000.pt'
N_RNN    = 256
SEED     = 0
N_TRIALS = 64
FIG_DIR  = os.path.join(RUN_DIR, 'figures', 'dynamics')
os.makedirs(FIG_DIR, exist_ok=True)

FAMILY_COLORS = {
    'delaypro'            : '#4C72B0',
    'delayanti'           : '#4C72B0',
    'memorypro'           : '#4C72B0',
    'memoryanti'          : '#4C72B0',
    'extendedmemory'      : '#4C72B0',
    'dm'                  : '#55A868',
    'dmanti'              : '#55A868',
    'contextdm_a'         : '#55A868',
    'contextdm_b'         : '#55A868',
    'delaymatchsample'    : '#8172B2',
    'delaynonmatchsample' : '#8172B2',
    'pulsecounting'       : '#C44E52',
    'intervalreproduction': '#C44E52',
    'pulserateestimation' : '#C44E52',
    'rhythmgeneration'    : '#DD8452',
    'sequencerecall'      : '#DD8452',
    'conditionalrhythm'   : '#DD8452',
    'toggle'              : '#937860',
    'conditionaltoggle'   : '#937860',
    'cueresponseassoc'    : '#DA8BC3',
    'pairedassociation'   : '#DA8BC3',
    'reversallearning'    : '#DA8BC3',
    'multiitemrecall'     : '#DA8BC3',
    'onlinelinearreg'     : '#8C8C8C',
    'onlinenonlinearreg'  : '#8C8C8C',
    'fewshotclassif'      : '#8C8C8C',
    'memorydm'            : '#64B5CD',
    'countandrecall'      : '#64B5CD',
    'delayedassociation'  : '#64B5CD',
    'sequentialdecision'  : '#64B5CD',
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

FAMILY_LEGEND = {
    'Memory'          : '#4C72B0',
    'Decision'        : '#55A868',
    'Match'           : '#8172B2',
    'Counting/Timing' : '#C44E52',
    'Rhythm/Seq'      : '#DD8452',
    'Bistable'        : '#937860',
    'Associative'     : '#DA8BC3',
    'In-Context'      : '#8C8C8C',
    'Compositional'   : '#64B5CD',
}

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

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

print('Loading model...')
all_tasks = {**YANG_DRISCOLL_TASKS, **NEW_TASKS}
config    = make_config(n_rnn=N_RNN, batch_size=N_TRIALS)
dataset   = TaskDataset(all_tasks, config, seed=SEED)

model = build_network(config)
ckpt  = torch.load(os.path.join(RUN_DIR, CKPT), map_location='cpu')
model.load_state_dict(ckpt['model_state'])
if 'task_vecs' in ckpt:
    dataset.task_vecs = ckpt['task_vecs']
model.eval()
print(f'Loaded {CKPT}')

# ---------------------------------------------------------------------------
# Extract hidden states for all tasks
# ---------------------------------------------------------------------------

def get_hidden_states(task_name, task_fn):
    cfg      = make_config(n_rnn=N_RNN, batch_size=N_TRIALS)
    rng      = np.random.RandomState(hash(task_name) % 2**31)
    task_vec = dataset.task_vecs[task_name]
    trial    = task_fn(cfg, rng)
    trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec
    trial.add_input_noise(rng)
    x_t, _, _ = trial.to_tensors()
    with torch.no_grad():
        result = model(x_t)
    return result.hidden.numpy(), trial.epochs   # (T, B, N_RNN), epochs


print('Extracting hidden states for all 30 tasks...')
task_hidden = {}
task_epochs = {}
for name, fn in all_tasks.items():
    h, epochs = get_hidden_states(name, fn)
    task_hidden[name] = h
    task_epochs[name] = epochs
    print(f'  {name:<25} T={h.shape[0]}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def endpoints(task_name):
    """Final hidden state per trial. Shape: (N_TRIALS, N_RNN)."""
    return task_hidden[task_name][-1, :, :]


def all_timesteps(task_name):
    """All hidden states flattened. Shape: (T*N_TRIALS, N_RNN)."""
    return task_hidden[task_name].reshape(-1, N_RNN)


def resp_timesteps(task_name):
    """Hidden states during response epoch flattened. Shape: (T_resp*N_TRIALS, N_RNN)."""
    h      = task_hidden[task_name]
    epochs = task_epochs[task_name]
    resp_key = 'go1' if 'go1' in epochs else list(epochs.keys())[-1]
    start, end = epochs[resp_key]
    return h[start:end, :, :].reshape(-1, N_RNN)


# ---------------------------------------------------------------------------
# Fit per-task PCA on response endpoints
# ---------------------------------------------------------------------------

task_pcas = {}
for name in all_tasks:
    pca = PCA(n_components=3)
    pca.fit(endpoints(name))
    task_pcas[name] = pca


# ---------------------------------------------------------------------------
# Figure 1: PCA trajectories for selected tasks
# ---------------------------------------------------------------------------

print('\nFigure 1: PCA trajectories...')

SELECTED = [
    'memorypro', 'memoryanti', 'dm',
    'pulsecounting', 'onlinelinearreg', 'sequentialdecision',
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
colors_trial = plt.cm.hsv(np.linspace(0, 1, N_TRIALS))

for idx, task_name in enumerate(SELECTED):
    ax     = axes[idx]
    h      = task_hidden[task_name]   # (T, B, N_RNN)
    epochs = task_epochs[task_name]

    # Fit PCA on all timepoints of this task.
    pca = PCA(n_components=2)
    pca.fit(h.reshape(-1, N_RNN))
    var = pca.explained_variance_ratio_

    resp_key = 'go1' if 'go1' in epochs else list(epochs.keys())[-1]
    resp_on  = epochs[resp_key][0]

    for b in range(min(N_TRIALS, 32)):
        traj = pca.transform(h[:, b, :])
        ax.plot(traj[:resp_on, 0], traj[:resp_on, 1],
                color='gray', alpha=0.15, linewidth=0.8)
        ax.plot(traj[resp_on:, 0], traj[resp_on:, 1],
                color=colors_trial[b], alpha=0.6, linewidth=1.0)

    # Endpoints only — larger markers.
    ep = pca.transform(endpoints(task_name))
    ax.scatter(ep[:, 0], ep[:, 1],
               c=colors_trial[:N_TRIALS], s=50, zorder=5,
               edgecolors='white', linewidth=0.5)

    ax.set_title(f'{TASK_LABELS.get(task_name, task_name)}\n'
                 f'PC1={var[0]:.1%}  PC2={var[1]:.1%}', fontsize=10)
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)

plt.suptitle('PCA Trajectories — Hidden State Dynamics\n'
             '30-Task LeakyRNN (256 units, 1M steps)\n'
             'Gray = pre-response  |  Color = response epoch  |  Dots = endpoints',
             fontsize=12, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_pca_trajectories.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Pro vs anti shared subspace
# ---------------------------------------------------------------------------

print('Figure 2: Pro vs anti shared subspace...')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, (task_pro, task_anti) in enumerate([
    ('memorypro', 'memoryanti'),
    ('delaypro',  'delayanti'),
]):
    ax    = axes[ax_idx]
    pca_a = task_pcas[task_pro]
    var   = pca_a.explained_variance_ratio_

    for task_name, marker, label, alpha in [
        (task_pro,  'o', TASK_LABELS.get(task_pro,  task_pro),  0.8),
        (task_anti, 's', TASK_LABELS.get(task_anti, task_anti), 0.8),
    ]:
        ep   = endpoints(task_name)
        proj = pca_a.transform(ep)
        # Color by angle of the projected point — proxy for stimulus angle.
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        colors = plt.cm.hsv((angles - angles.min()) /
                             (angles.max() - angles.min() + 1e-8))
        ax.scatter(proj[:, 0], proj[:, 1],
                   c=colors, marker=marker, s=70, alpha=alpha,
                   edgecolors='white', linewidth=0.5, label=label, zorder=5)

    ax.set_title(f'{TASK_LABELS.get(task_pro, task_pro)} vs '
                 f'{TASK_LABELS.get(task_anti, task_anti)}\n'
                 f'PCA fitted on {TASK_LABELS.get(task_pro, task_pro)}  '
                 f'PC1={var[0]:.1%}  PC2={var[1]:.1%}',
                 fontsize=10)
    ax.set_xlabel('PC1 (fitted on pro task)', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

plt.suptitle('Shared Subspace: Pro vs Anti Tasks\n'
             'Anti endpoints on pro PCA axes — same ring, different readout?',
             fontsize=12, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_pro_anti_shared_subspace.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Cross-task variance matrix (30x30)
# ---------------------------------------------------------------------------

print('Figure 3: Cross-task variance matrix...')

task_names    = list(all_tasks.keys())
n_tasks       = len(task_names)
var_matrix    = np.zeros((n_tasks, n_tasks))

for i, name_a in enumerate(task_names):
    pca_a = task_pcas[name_a]
    for j, name_b in enumerate(task_names):
        ep_b      = endpoints(name_b)
        proj      = pca_a.transform(ep_b)
        var_total = np.var(ep_b, axis=0).sum()
        var_proj  = np.var(proj, axis=0).sum()
        var_matrix[i, j] = var_proj / (var_total + 1e-8)

fig, ax = plt.subplots(figsize=(14, 12))
labels  = [TASK_LABELS.get(t, t) for t in task_names]

im = ax.imshow(var_matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Fraction variance explained', shrink=0.6)
ax.set_xticks(range(n_tasks))
ax.set_yticks(range(n_tasks))
ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('Task B (data being explained)', fontsize=11)
ax.set_ylabel('Task A (PCA fitted on A, applied to B)', fontsize=11)
ax.set_title('Cross-Task Variance Explained\n'
             'Entry (A,B) = fraction of task B endpoint variance\n'
             'explained by top 3 PCs of task A  (Driscoll Fig 4d equivalent)',
             fontsize=11)

# Family separator lines.
count = 0
for fam in FAMILIES_ORDERED:
    n = len([t for t in fam if t in task_names])
    count += n
    if count < n_tasks:
        ax.axhline(count - 0.5, color='white', linewidth=2)
        ax.axvline(count - 0.5, color='white', linewidth=2)

plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_cross_task_variance.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Unit variance matrix
# ---------------------------------------------------------------------------

print('Figure 4: Unit variance matrix...')

unit_var = np.zeros((N_RNN, n_tasks))
for j, name in enumerate(task_names):
    ep = endpoints(name)   # (N_TRIALS, N_RNN)
    unit_var[:, j] = np.var(ep, axis=0)

unit_var_norm = unit_var / (unit_var.max(axis=0, keepdims=True) + 1e-8)

clustering  = AgglomerativeClustering(n_clusters=min(8, N_RNN))
unit_labels = clustering.fit_predict(unit_var_norm)
unit_order  = np.argsort(unit_labels)

fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                          gridspec_kw={'width_ratios': [3, 1]})

ax = axes[0]
im = ax.imshow(unit_var_norm[unit_order], aspect='auto',
               cmap='hot', vmin=0, vmax=1, interpolation='nearest')
plt.colorbar(im, ax=ax, label='Normalized variance', shrink=0.6)
ax.set_xticks(range(n_tasks))
ax.set_xticklabels([TASK_LABELS.get(t, t) for t in task_names],
                   rotation=90, fontsize=7)
ax.set_ylabel('Units (sorted by clustering)', fontsize=10)
ax.set_title('Unit Variance Matrix\n(normalized per task, sorted by clustering)',
             fontsize=10)

ax = axes[1]
mean_var = unit_var_norm.mean(axis=0)
colors   = [FAMILY_COLORS.get(t, '#888888') for t in task_names]
ax.barh(range(n_tasks), mean_var[::-1], color=colors[::-1])
ax.set_yticks(range(n_tasks))
ax.set_yticklabels([TASK_LABELS.get(t, t) for t in task_names[::-1]], fontsize=7)
ax.set_xlabel('Mean normalized variance', fontsize=9)
ax.set_title('Mean unit\nvariance per task', fontsize=10)

plt.suptitle('Unit Selectivity Across Tasks  (Driscoll Fig 3a equivalent)',
             fontsize=12, y=1.01)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_variance_matrix.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 5: Global PCA — all tasks in one space
# ---------------------------------------------------------------------------

print('Figure 5: Global PCA...')

# Fit PCA on all task endpoints stacked.
all_ep     = np.concatenate([endpoints(t) for t in task_names], axis=0)
pca_global = PCA(n_components=3)
pca_global.fit(all_ep)
var = pca_global.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (pc_x, pc_y) in zip(axes, [(0, 1), (0, 2)]):
    for name in task_names:
        ep    = endpoints(name)
        proj  = pca_global.transform(ep)
        color = FAMILY_COLORS.get(name, '#888888')
        ax.scatter(proj[:, pc_x], proj[:, pc_y],
                   color=color, alpha=0.5, s=20)
        cx = proj[:, pc_x].mean()
        cy = proj[:, pc_y].mean()
        ax.annotate(TASK_LABELS.get(name, name)[:8],
                    (cx, cy), fontsize=5, color=color,
                    ha='center', va='center', fontweight='bold')

    ax.set_xlabel(f'PC{pc_x+1} ({var[pc_x]:.1%})', fontsize=10)
    ax.set_ylabel(f'PC{pc_y+1} ({var[pc_y]:.1%})', fontsize=10)
    ax.set_title(f'Global PCA: PC{pc_x+1} vs PC{pc_y+1}', fontsize=11)
    ax.grid(True, alpha=0.2)

legend_elements = [Patch(facecolor=c, label=n)
                   for n, c in FAMILY_LEGEND.items()]
axes[0].legend(handles=legend_elements, fontsize=7,
               loc='upper right', ncol=2)

plt.suptitle('Global PCA: All 30 Tasks — Endpoint Distribution\n'
             '(colored by task family, labeled by task)',
             fontsize=12, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_global_pca.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 6: Compositional task subspace
# ---------------------------------------------------------------------------

print('Figure 6: Compositional subspace...')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

comp_pairs = [
    ('memorydm',          'memorypro', 'dm',           'MemoryDM vs components'),
    ('countandrecall',    'memorypro', 'pulsecounting', 'CountRecall vs components'),
    ('sequentialdecision','dm',        'memorypro',     'SeqDec vs DM + Memory'),
]

for ax_idx, (comp_task, comp_a, comp_b, title) in enumerate(comp_pairs):
    ax    = axes[ax_idx]
    pca_a = task_pcas[comp_a]
    var   = pca_a.explained_variance_ratio_

    for task_name, marker, label, size in [
        (comp_a,    'o', TASK_LABELS.get(comp_a,    comp_a),    60),
        (comp_b,    's', TASK_LABELS.get(comp_b,    comp_b),    60),
        (comp_task, '^', TASK_LABELS.get(comp_task, comp_task), 80),
    ]:
        ep    = endpoints(task_name)
        proj  = pca_a.transform(ep)
        color = FAMILY_COLORS.get(task_name, '#888888')
        ax.scatter(proj[:, 0], proj[:, 1],
                   color=color, marker=marker, s=size,
                   alpha=0.75, label=label,
                   edgecolors='white', linewidth=0.5, zorder=5)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f'PC1 of {TASK_LABELS.get(comp_a, comp_a)}\n'
                  f'({var[0]:.1%} var)', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

plt.suptitle('Compositional Task Subspace Analysis\n'
             'Trial endpoints — do compositional tasks overlap components?',
             fontsize=12, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_compositional_subspace.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 7: Memory task ring — response epoch trajectory colored by time
# ---------------------------------------------------------------------------

print('Figure 7: Memory ring detail...')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ring_tasks = ['memorypro', 'memoryanti', 'delaypro']

for ax_idx, task_name in enumerate(ring_tasks):
    ax     = axes[ax_idx]
    h      = task_hidden[task_name]
    epochs = task_epochs[task_name]

    pca = PCA(n_components=2)
    pca.fit(h.reshape(-1, N_RNN))
    var = pca.explained_variance_ratio_

    resp_key = 'go1' if 'go1' in epochs else list(epochs.keys())[-1]
    resp_on  = epochs[resp_key][0]
    stim_key = 'stim1' if 'stim1' in epochs else None

    colors_trial = plt.cm.hsv(np.linspace(0, 1, N_TRIALS))

    for b in range(N_TRIALS):
        traj = pca.transform(h[:, b, :])
        # Pre-stim: very faint gray
        if stim_key:
            stim_on = epochs[stim_key][0]
            ax.plot(traj[:stim_on, 0], traj[:stim_on, 1],
                    color='lightgray', alpha=0.1, linewidth=0.5)
            ax.plot(traj[stim_on:resp_on, 0], traj[stim_on:resp_on, 1],
                    color='gray', alpha=0.3, linewidth=0.8)
        else:
            ax.plot(traj[:resp_on, 0], traj[:resp_on, 1],
                    color='gray', alpha=0.2, linewidth=0.8)
        # Response in color.
        ax.plot(traj[resp_on:, 0], traj[resp_on:, 1],
                color=colors_trial[b], alpha=0.7, linewidth=1.2)

    # Endpoints — large dots.
    ep = pca.transform(endpoints(task_name))
    ax.scatter(ep[:, 0], ep[:, 1],
               c=colors_trial[:N_TRIALS], s=80, zorder=6,
               edgecolors='black', linewidth=0.5)

    ax.set_title(f'{TASK_LABELS.get(task_name, task_name)}\n'
                 f'PC1={var[0]:.1%}  PC2={var[1]:.1%}', fontsize=11)
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=9)

plt.suptitle('Memory Task Ring Attractor Detail\n'
             'Light gray = fixation  |  Gray = delay  |  '
             'Color = response (HSV = stimulus angle)',
             fontsize=12, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_memory_ring_detail.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('DYNAMICS ANALYSIS COMPLETE')
print('=' * 60)
print(f'\nFigures saved to: {FIG_DIR}')
print('\nFigures generated:')
for fig_name, desc in [
    ('fig_pca_trajectories.png',         'PCA trajectories for 6 selected tasks'),
    ('fig_pro_anti_shared_subspace.png', 'Pro vs anti ring attractor test'),
    ('fig_cross_task_variance.png',      '30x30 cross-task variance matrix'),
    ('fig_variance_matrix.png',          'Unit selectivity matrix'),
    ('fig_global_pca.png',               'All 30 tasks in global PCA space'),
    ('fig_compositional_subspace.png',   'Compositional task subspace'),
    ('fig_memory_ring_detail.png',       'Memory ring attractor detail'),
]:
    print(f'  {fig_name:<40} {desc}')

print('\nKey things to look for:')
print('  fig_memory_ring_detail: endpoints should form a ring ordered by angle')
print('  fig_pro_anti_shared_subspace: anti ring should mirror pro ring')
print('  fig_cross_task_variance: block structure along diagonal by family')
print('  fig_compositional_subspace: comp task endpoints overlap components?')