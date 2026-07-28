"""
Tasks analyzed: toggle, conditionaltoggle, rhythmgeneration, conditionalrhythm

For each task and architecture, shows:
    - Output traces vs target (multiple trials overlaid)
    - MSE over time within the response/stream period
    - Amplitude envelope (for rhythm tasks)
    - Output distribution (for toggle tasks)

Usage:
    python analysis/analyze_failed_tasks_comparison.py

Saves figures to:
    analysis/figures/failed_task_comparison/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, '.')

from tasks.base import make_config, TaskDataset, IN_CUE_0, IN_CUE_3
from tasks.yang_driscoll import YANG_DRISCOLL_TASKS
from tasks.new_tasks import NEW_TASKS
from network.rnn import build_network
from network.train import compute_performance, masked_mse_loss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARCHITECTURES = [
    {
        'name'    : 'LeakyRNN',
        'run_dir' : 'runs/LeakyRNN_256units_30tasks_seed0',
        'ckpt'    : 'ckpt_step1000000.pt',
        'rnn_type': 'LeakyRNN',
        'n_rnn'   : 256,
        'color'   : '#4C72B0',
    },
    {
        'name'    : 'GRU',
        'run_dir' : 'runs/GRU_256units_30tasks_seed0',
        'ckpt'    : 'ckpt_step4300000.pt',
        'rnn_type': 'GRU',
        'n_rnn'   : 256,
        'color'   : '#55A868',
    },
    {
        'name'    : 'Transformer',
        'run_dir' : 'runs/Transformer_128d_30tasks_seed0',
        'ckpt'    : 'ckpt_step5700000.pt',
        'rnn_type': 'Transformer',
        'n_rnn'   : 128,
        'color'   : '#C44E52',
    },
]

FAILED_TASKS = {
    'rhythmgeneration' : {
        'out_ch'    : 3,
        'type'      : 'rhythm',
        'label'     : 'Rhythm Generation',
        'resp_epoch': 'sustain',
    },
    'conditionalrhythm': {
        'out_ch'    : 3,
        'type'      : 'rhythm',
        'label'     : 'Conditional Rhythm',
        'resp_epoch': 'phase_1',
    },
    'toggle'           : {
        'out_ch'    : 3,
        'type'      : 'toggle',
        'label'     : 'Toggle',
        'resp_epoch': 'stream',
    },
    'conditionaltoggle': {
        'out_ch'    : 3,
        'type'      : 'toggle',
        'label'     : 'Conditional Toggle',
        'resp_epoch': 'stream',
    },
}

SEED    = 0
N_TRIALS= 8
FIG_DIR = 'analysis/figures/failed_task_comparison'
os.makedirs(FIG_DIR, exist_ok=True)

all_tasks = {**YANG_DRISCOLL_TASKS, **NEW_TASKS}

# ---------------------------------------------------------------------------
# Load all models
# ---------------------------------------------------------------------------

print('Loading models...')
models  = {}
datasets= {}

for arch in ARCHITECTURES:
    config  = make_config(n_rnn=arch['n_rnn'], batch_size=N_TRIALS,
                          rnn_type=arch['rnn_type'])
    dataset = TaskDataset(all_tasks, config, seed=SEED)
    model   = build_network(config)

    ckpt_path = os.path.join(arch['run_dir'], arch['ckpt'])
    ckpt      = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if 'task_vecs' in ckpt:
        dataset.task_vecs = ckpt['task_vecs']
    model.eval()

    models[arch['name']]   = model
    datasets[arch['name']] = dataset
    print(f'  Loaded {arch["name"]} from {arch["ckpt"]}')


# ---------------------------------------------------------------------------
# Helper: run one task for one architecture
# ---------------------------------------------------------------------------

def run_task(arch_name, task_name, n_trials=N_TRIALS, seed=0):
    arch     = next(a for a in ARCHITECTURES if a['name'] == arch_name)
    model    = models[arch_name]
    dataset  = datasets[arch_name]
    config   = make_config(n_rnn=arch['n_rnn'], batch_size=n_trials,
                           rnn_type=arch['rnn_type'])
    rng      = np.random.RandomState(seed)
    task_vec = dataset.task_vecs[task_name]
    fn       = all_tasks[task_name]

    trial = fn(config, rng)
    trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec
    trial.add_input_noise(rng)
    x_t, y_t, c_t = trial.to_tensors()

    with torch.no_grad():
        result = model(x_t)

    return {
        'output': result.output.numpy(),  # (T, B, 5)
        'y'     : y_t.numpy(),
        'epochs': trial.epochs,
        'T'     : trial.tdim,
    }


# ---------------------------------------------------------------------------
# Figure 1: Output traces side by side for each failed task
# One row per architecture, one column per task
# ---------------------------------------------------------------------------

print('\nFigure 1: Output traces comparison...')

fig, axes = plt.subplots(
    len(ARCHITECTURES), len(FAILED_TASKS),
    figsize=(5 * len(FAILED_TASKS), 4 * len(ARCHITECTURES))
)

for row_idx, arch in enumerate(ARCHITECTURES):
    for col_idx, (task_name, task_info) in enumerate(FAILED_TASKS.items()):
        ax      = axes[row_idx, col_idx]
        data    = run_task(arch['name'], task_name)
        out_ch  = task_info['out_ch']
        t       = np.arange(data['T'])
        color   = arch['color']
        epochs  = data['epochs']

        # Plot target for trial 0 (thick black dashed)
        ax.plot(t, data['y'][:, 0, out_ch],
                linestyle='--', color='black', linewidth=2.0,
                label='target', zorder=10)

        # Overlay multiple trials of network output
        alphas = np.linspace(0.3, 0.8, N_TRIALS)
        for b in range(N_TRIALS):
            ax.plot(t, data['output'][:, b, out_ch],
                    color=color, alpha=alphas[b], linewidth=0.9)

        # Mark epoch boundaries
        for ep_name, (ep_start, ep_end) in epochs.items():
            if ep_start > 0:
                ax.axvline(ep_start, color='gray',
                           linestyle=':', linewidth=0.8)
            ep_end_t = ep_end if ep_end else data['T']
            ax.text((ep_start + ep_end_t) / 2,
                    ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
                    ep_name, fontsize=6, ha='center',
                    va='top', color='gray')

        # Title only on top row
        if row_idx == 0:
            ax.set_title(task_info['label'], fontsize=10, fontweight='bold')

        # Arch label only on left column
        if col_idx == 0:
            ax.set_ylabel(arch['name'], fontsize=10,
                          color=color, fontweight='bold')

        ax.set_xlim([0, data['T']])
        ax.set_ylim([-1.5, 1.5])
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

        if row_idx == len(ARCHITECTURES) - 1:
            ax.set_xlabel('Timestep', fontsize=9)

plt.suptitle('Failed Tasks: Output Traces Across Architectures\n'
             'Black dashed = target  |  Colored = network output (8 trials)',
             fontsize=13, y=1.01)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_failed_traces_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 2: MSE over time for rhythm tasks — all architectures overlaid
# ---------------------------------------------------------------------------

print('Figure 2: MSE over time for rhythm tasks...')

rhythm_tasks = {k: v for k, v in FAILED_TASKS.items() if v['type'] == 'rhythm'}

fig, axes = plt.subplots(1, len(rhythm_tasks), figsize=(7 * len(rhythm_tasks), 5))
if len(rhythm_tasks) == 1:
    axes = [axes]

for col_idx, (task_name, task_info) in enumerate(rhythm_tasks.items()):
    ax          = axes[col_idx]
    resp_epoch  = task_info['resp_epoch']

    for arch in ARCHITECTURES:
        data   = run_task(arch['name'], task_name, n_trials=16)
        out_ch = task_info['out_ch']
        epochs = data['epochs']

        if resp_epoch not in epochs:
            resp_epoch_key = list(epochs.keys())[-1]
        else:
            resp_epoch_key = resp_epoch

        resp_on  = epochs[resp_epoch_key][0]
        resp_end = epochs[resp_epoch_key][1] or data['T']

        resp_out = data['output'][resp_on:resp_end, :, out_ch]
        resp_tgt = data['y'][resp_on:resp_end, :, out_ch]
        mse_t    = ((resp_out - resp_tgt) ** 2).mean(axis=1)
        t_axis   = np.arange(len(mse_t))

        ax.plot(t_axis, mse_t, color=arch['color'],
                linewidth=2, alpha=0.85, label=arch['name'])

    ax.axhline(0.05, color='black', linestyle='--',
               linewidth=1.2, label='Threshold (0.05)')
    ax.set_title(f'{task_info["label"]}\nMSE over {resp_epoch} period',
                 fontsize=11)
    ax.set_xlabel(f'Timesteps into {resp_epoch} period', fontsize=10)
    ax.set_ylabel('MSE (mean over 16 trials)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.02, 1.0])

plt.suptitle('Rhythm Tasks: MSE over Time — All Architectures\n'
             'Do GRU/Transformer sustain oscillation longer than LeakyRNN?',
             fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_rhythm_mse_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Amplitude envelope for rhythm tasks
# ---------------------------------------------------------------------------

print('Figure 3: Amplitude envelopes...')

fig, axes = plt.subplots(1, len(rhythm_tasks), figsize=(7 * len(rhythm_tasks), 5))
if len(rhythm_tasks) == 1:
    axes = [axes]

for col_idx, (task_name, task_info) in enumerate(rhythm_tasks.items()):
    ax         = axes[col_idx]
    resp_epoch = task_info['resp_epoch']
    window     = 15

    for arch in ARCHITECTURES:
        data   = run_task(arch['name'], task_name, n_trials=16)
        out_ch = task_info['out_ch']
        epochs = data['epochs']

        resp_epoch_key = resp_epoch if resp_epoch in epochs else list(epochs.keys())[-1]
        resp_on  = epochs[resp_epoch_key][0]
        resp_end = epochs[resp_epoch_key][1] or data['T']

        resp_out = data['output'][resp_on:resp_end, :, out_ch]
        resp_tgt = data['y'][resp_on:resp_end, :, out_ch]
        T_resp   = resp_out.shape[0]

        amp_out = np.array([
            np.abs(resp_out[max(0, i-window):i+window]).max(axis=0).mean()
            for i in range(T_resp)
        ])
        amp_tgt = np.array([
            np.abs(resp_tgt[max(0, i-window):i+window]).max(axis=0).mean()
            for i in range(T_resp)
        ])

        t_axis = np.arange(T_resp)
        ax.plot(t_axis, amp_out, color=arch['color'],
                linewidth=2, alpha=0.85, label=arch['name'])

    # Target amplitude (same for all architectures)
    ax.plot(t_axis, amp_tgt, color='black', linewidth=2,
            linestyle='--', label='Target', zorder=10)

    ax.set_title(f'{task_info["label"]}\nAmplitude Envelope', fontsize=11)
    ax.set_xlabel(f'Timesteps into {resp_epoch} period', fontsize=10)
    ax.set_ylabel('Amplitude (rolling max)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.3])

plt.suptitle('Rhythm Tasks: Amplitude Decay — All Architectures\n'
             'Which architecture sustains oscillation longest?',
             fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_rhythm_amplitude_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Toggle output distribution — all architectures
# ---------------------------------------------------------------------------

print('Figure 4: Toggle distributions...')

toggle_tasks = {k: v for k, v in FAILED_TASKS.items() if v['type'] == 'toggle'}

fig, axes = plt.subplots(
    len(ARCHITECTURES), len(toggle_tasks),
    figsize=(5 * len(toggle_tasks), 4 * len(ARCHITECTURES))
)
if len(toggle_tasks) == 1:
    axes = axes.reshape(-1, 1)

for row_idx, arch in enumerate(ARCHITECTURES):
    for col_idx, (task_name, task_info) in enumerate(toggle_tasks.items()):
        ax         = axes[row_idx, col_idx]
        data       = run_task(arch['name'], task_name, n_trials=16)
        out_ch     = task_info['out_ch']
        resp_epoch = task_info['resp_epoch']
        epochs     = data['epochs']

        resp_on  = epochs[resp_epoch][0] if resp_epoch in epochs else 0
        resp_end = epochs[resp_epoch][1] or data['T'] if resp_epoch in epochs else data['T']

        stream_out = data['output'][resp_on:resp_end, :, out_ch].flatten()
        stream_tgt = data['y'][resp_on:resp_end, :, out_ch].flatten()

        ax.hist(stream_tgt, bins=40, alpha=0.5,
                color='gray', label='Target', density=True)
        ax.hist(stream_out, bins=40, alpha=0.7,
                color=arch['color'], label=f'{arch["name"]} output',
                density=True)

        out_std = stream_out.std()
        ax.set_title(
            f'{task_info["label"]} — {arch["name"]}\n'
            f'Output std={out_std:.4f}  '
            f'{"(flat zero)" if out_std < 0.05 else "(has variance)"}',
            fontsize=9,
            color='red' if out_std < 0.05 else arch['color'])
        ax.set_xlabel('Activation value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

plt.suptitle('Toggle Tasks: Output Distribution — All Architectures\n'
             'Target = bimodal at ±1  |  Output should also be bimodal',
             fontsize=13, y=1.01)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_toggle_distribution_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 5: Early window performance for rhythm tasks — all architectures
# ---------------------------------------------------------------------------

print('Figure 5: Early window performance...')

fig, axes = plt.subplots(1, len(rhythm_tasks), figsize=(7 * len(rhythm_tasks), 5))
if len(rhythm_tasks) == 1:
    axes = [axes]

windows = [5, 10, 20, 30, 50, 75, 100, 150]

for col_idx, (task_name, task_info) in enumerate(rhythm_tasks.items()):
    ax         = axes[col_idx]
    resp_epoch = task_info['resp_epoch']

    for arch in ARCHITECTURES:
        data   = run_task(arch['name'], task_name, n_trials=16)
        out_ch = task_info['out_ch']
        epochs = data['epochs']

        resp_epoch_key = resp_epoch if resp_epoch in epochs else list(epochs.keys())[-1]
        resp_on  = epochs[resp_epoch_key][0]
        resp_end = epochs[resp_epoch_key][1] or data['T']

        resp_out = data['output'][resp_on:resp_end, :, out_ch]
        resp_tgt = data['y'][resp_on:resp_end, :, out_ch]

        early_perfs = []
        for w in windows:
            w = min(w, resp_out.shape[0])
            trial_mses   = ((resp_out[:w] - resp_tgt[:w])**2).mean(axis=0)
            frac_correct = (trial_mses < 0.05).mean()
            early_perfs.append(frac_correct)

        ax.plot(windows, early_perfs, 'o-', color=arch['color'],
                linewidth=2, markersize=7, label=arch['name'], alpha=0.85)

        # Annotate first window value
        ax.annotate(f'{early_perfs[0]:.2f}',
                    (windows[0], early_perfs[0]),
                    textcoords='offset points', xytext=(5, 5),
                    fontsize=8, color=arch['color'])

    ax.axhline(0.9, color='gray', linestyle='--',
               linewidth=1, label='Solved (0.9)')
    ax.set_title(f'{task_info["label"]}\nEarly Window Performance',
                 fontsize=11)
    ax.set_xlabel('Evaluation window (timesteps into response)', fontsize=10)
    ax.set_ylabel('Fraction correct (MSE < 0.05)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

plt.suptitle('Rhythm Tasks: Early Window Performance — All Architectures\n'
             'Higher early performance = oscillation initiates correctly\n'
             'Faster decay = less stable limit cycle',
             fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_rhythm_early_window_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------

print('\n' + '=' * 65)
print('FAILED TASK COMPARISON SUMMARY')
print('=' * 65)

for task_name, task_info in FAILED_TASKS.items():
    print(f'\n{task_info["label"]}  ({task_info["type"]}):')
    resp_epoch = task_info['resp_epoch']
    out_ch     = task_info['out_ch']

    for arch in ARCHITECTURES:
        data   = run_task(arch['name'], task_name, n_trials=16)
        epochs = data['epochs']

        resp_epoch_key = resp_epoch if resp_epoch in epochs else list(epochs.keys())[-1]
        resp_on  = epochs[resp_epoch_key][0]
        resp_end = epochs[resp_epoch_key][1] or data['T']

        resp_out = data['output'][resp_on:resp_end, :, out_ch]
        resp_tgt = data['y'][resp_on:resp_end, :, out_ch]

        out_std  = resp_out.std()
        full_mse = ((resp_out - resp_tgt)**2).mean()

        if task_info['type'] == 'rhythm':
            w = min(10, resp_out.shape[0])
            early_mses   = ((resp_out[:w] - resp_tgt[:w])**2).mean(axis=0)
            early_correct = (early_mses < 0.05).mean()
            print(f'  {arch["name"]:<12} std={out_std:.4f}  '
                  f'full_mse={full_mse:.3f}  '
                  f'early_correct(10ts)={early_correct:.2f}')
        else:
            print(f'  {arch["name"]:<12} std={out_std:.5f}  '
                  f'full_mse={full_mse:.3f}  '
                  f'{"FLAT ZERO" if out_std < 0.02 else "HAS VARIANCE"}')

print(f'\nFigures saved to: {FIG_DIR}')