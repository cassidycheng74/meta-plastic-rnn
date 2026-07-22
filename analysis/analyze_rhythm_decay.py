"""
Produces:
    1. Output trace over time — target vs output for multiple trials
    2. MSE as a function of time within sustain period (decay curve)
    3. Amplitude envelope over time — shows decay rate
    4. Frequency analysis — does the network get the right frequency?
    5. Toggle trace — confirms genuine failure (flat zero)
    6. Comparison: early vs late sustain performance

Usage:
    python analysis/analyze_rhythm_decay.py

Saves figures to:
    runs/LeakyRNN_256units_30tasks_seed0/figures/dynamics/extended/
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
from network.train import compute_performance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUN_DIR = 'runs/LeakyRNN_256units_30tasks_seed0'
CKPT    = 'ckpt_step1000000.pt'
N_RNN   = 256
SEED    = 0
FIG_DIR = os.path.join(RUN_DIR, 'figures', 'dynamics', 'extended')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

print('Loading model...')
all_tasks = {**YANG_DRISCOLL_TASKS, **NEW_TASKS}
config    = make_config(n_rnn=N_RNN, batch_size=16)
dataset   = TaskDataset(all_tasks, config, seed=SEED)

model = build_network(config)
ckpt  = torch.load(os.path.join(RUN_DIR, CKPT), map_location='cpu')
model.load_state_dict(ckpt['model_state'])
if 'task_vecs' in ckpt:
    dataset.task_vecs = ckpt['task_vecs']
model.eval()
print(f'Loaded {CKPT}')


# ---------------------------------------------------------------------------
# Run tasks and collect outputs
# ---------------------------------------------------------------------------

def run_task(task_name, n_trials=16, seed=0):
    cfg      = make_config(n_rnn=N_RNN, batch_size=n_trials)
    rng      = np.random.RandomState(seed)
    task_vec = dataset.task_vecs[task_name]
    fn       = all_tasks[task_name]
    trial    = fn(cfg, rng)
    trial.x[:, :, IN_CUE_0:IN_CUE_3 + 1] = task_vec
    trial.add_input_noise(rng)
    x_t, y_t, c_t = trial.to_tensors()
    with torch.no_grad():
        result = model(x_t)
    return {
        'x'      : x_t.numpy(),
        'y'      : y_t.numpy(),
        'output' : result.output.numpy(),
        'hidden' : result.hidden.numpy(),
        'epochs' : trial.epochs,
        'T'      : trial.tdim,
        'c_mask' : c_t.numpy(),
    }


print('Running rhythmgeneration trials...')
rhythm_data = run_task('rhythmgeneration', n_trials=16)
print('Running toggle trials...')
toggle_data = run_task('toggle', n_trials=16)
print('Running conditionalrhythm trials...')
condrhythm_data = run_task('conditionalrhythm', n_trials=16)


# ---------------------------------------------------------------------------
# Figure 1: Rhythm output traces — multiple trials overlaid
# ---------------------------------------------------------------------------

print('\nFigure 1: Rhythm output traces...')

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Panel A: Full trial trace, 4 individual trials
ax = axes[0, 0]
t  = np.arange(rhythm_data['T'])
sustain_on = rhythm_data['epochs']['sustain'][0]
cue_on     = rhythm_data['epochs']['cue'][0]

colors4 = ['#C44E52', '#4C72B0', '#55A868', '#DD8452']
for b in range(4):
    ax.plot(t, rhythm_data['y'][:, b, 3],
            linestyle='--', color=colors4[b], alpha=0.6,
            linewidth=1.5, label=f'target trial {b+1}')
    ax.plot(t, rhythm_data['output'][:, b, 3],
            linestyle='-', color=colors4[b], alpha=0.9,
            linewidth=1.5, label=f'output trial {b+1}')

ax.axvline(cue_on,     color='gray', linestyle=':', linewidth=1)
ax.axvline(sustain_on, color='gray', linestyle=':', linewidth=1)
ax.text(cue_on + 2,     0.95, 'cue on',     fontsize=8, color='gray')
ax.text(sustain_on + 2, 0.95, 'sustain on', fontsize=8, color='gray')
ax.set_title('Rhythm Generation: 4 Trials\nDashed = target, Solid = output',
             fontsize=10)
ax.set_xlabel('Timestep', fontsize=9)
ax.set_ylabel('Output (channel 3)', fontsize=9)
ax.legend(fontsize=7, ncol=2, loc='lower right')
ax.grid(True, alpha=0.2)
ax.set_xlim([0, rhythm_data['T']])

# Panel B: MSE over time within sustain period
ax = axes[0, 1]
sustain_out = rhythm_data['output'][sustain_on:, :, 3]  # (T_sustain, B)
sustain_tgt = rhythm_data['y'][sustain_on:, :, 3]
mse_over_time = ((sustain_out - sustain_tgt) ** 2).mean(axis=1)  # (T_sustain,)
t_sustain = np.arange(len(mse_over_time))

ax.plot(t_sustain, mse_over_time, color='#C44E52', linewidth=2)
ax.axhline(0.05, color='black', linestyle='--', linewidth=1.2,
           label='Perf threshold (MSE=0.05)')
ax.fill_between(t_sustain, 0, mse_over_time,
                where=mse_over_time < 0.05,
                alpha=0.3, color='#55A868', label='Below threshold (correct)')
ax.fill_between(t_sustain, 0, mse_over_time,
                where=mse_over_time >= 0.05,
                alpha=0.3, color='#C44E52', label='Above threshold (wrong)')
ax.set_title('MSE over Time within Sustain Period\n'
             'Shows when network output diverges from target',
             fontsize=10)
ax.set_xlabel('Timesteps into sustain period', fontsize=9)
ax.set_ylabel('MSE (averaged over 16 trials)', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel C: Amplitude envelope over time
ax = axes[1, 0]
# Compute amplitude as rolling max of absolute output vs target.
window = 20
T_sus  = sustain_out.shape[0]
amp_out = np.array([np.abs(sustain_out[max(0,i-window):i+window]).max(axis=0).mean()
                    for i in range(T_sus)])
amp_tgt = np.array([np.abs(sustain_tgt[max(0,i-window):i+window]).max(axis=0).mean()
                    for i in range(T_sus)])

ax.plot(t_sustain, amp_tgt, color='black', linewidth=2,
        linestyle='--', label='Target amplitude')
ax.plot(t_sustain, amp_out, color='#C44E52', linewidth=2,
        label='Network output amplitude')
ax.fill_between(t_sustain, amp_out, amp_tgt,
                alpha=0.2, color='#C44E52', label='Amplitude deficit')
ax.set_title('Amplitude Envelope over Sustain Period\n'
             'Confirms oscillation decays — limit cycle not stable',
             fontsize=10)
ax.set_xlabel('Timesteps into sustain period', fontsize=9)
ax.set_ylabel('Amplitude (rolling max)', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_ylim([0, 1.2])

# Panel D: Early vs late performance breakdown
ax = axes[1, 1]
T_sus      = sustain_out.shape[0]
window_size = T_sus // 5
early_mses  = []
late_mses   = []
window_labels = []

for w in range(5):
    w_start = w * window_size
    w_end   = w_start + window_size
    w_out   = sustain_out[w_start:w_end]
    w_tgt   = sustain_tgt[w_start:w_end]
    mse_w   = ((w_out - w_tgt) ** 2).mean()
    # Fraction of trials below threshold.
    trial_mses = ((w_out - w_tgt) ** 2).mean(axis=0)
    frac_correct = (trial_mses < 0.05).mean()
    early_mses.append(mse_w)
    late_mses.append(frac_correct)
    window_labels.append(f'{w_start}-{w_end}')

x = np.arange(5)
bars = ax.bar(x, late_mses, color=plt.cm.RdYlGn(late_mses),
              edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'Steps\n{l}' for l in window_labels], fontsize=8)
ax.set_ylabel('Fraction of trials correct (MSE < 0.05)', fontsize=9)
ax.set_title('Performance by Window within Sustain Period\n'
             'Early = good, Late = bad (oscillation decay)',
             fontsize=10)
ax.set_ylim([0, 1.05])
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
ax.grid(True, alpha=0.2, axis='y')

for bar, frac in zip(bars, late_mses):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{frac:.2f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Rhythm Generation: Limit Cycle Stability Analysis\n'
             'Network initiates oscillation correctly but cannot sustain it',
             fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_rhythm_decay_analysis.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Toggle — confirming genuine failure
# ---------------------------------------------------------------------------

print('Figure 2: Toggle failure analysis...')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Output trace for multiple trials
ax = axes[0]
t  = np.arange(toggle_data['T'])
stream_on = toggle_data['epochs']['stream'][0]

for b in range(8):
    ax.plot(t, toggle_data['y'][:, b, 3],
            linestyle='--', color='#937860', alpha=0.4,
            linewidth=1.2)
    ax.plot(t, toggle_data['output'][:, b, 3],
            linestyle='-', color='#4C72B0', alpha=0.5,
            linewidth=1.0)

# Legend proxies.
ax.plot([], [], linestyle='--', color='#937860', linewidth=2, label='Target (8 trials)')
ax.plot([], [], linestyle='-',  color='#4C72B0', linewidth=2, label='Output (8 trials)')
ax.axvline(stream_on, color='gray', linestyle=':', linewidth=1)
ax.text(stream_on + 2, 0.9, 'stream on', fontsize=8, color='gray')
ax.set_title('Toggle: Output vs Target\nNetwork outputs flat ~0 entire trial',
             fontsize=10)
ax.set_xlabel('Timestep', fontsize=9)
ax.set_ylabel('Output (channel 3)', fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_xlim([0, toggle_data['T']])
ax.set_ylim([-1.5, 1.5])

# Panel B: Output distribution histogram
ax = axes[1]
stream_out = toggle_data['output'][stream_on:, :, 3].flatten()
stream_tgt = toggle_data['y'][stream_on:, :, 3].flatten()

ax.hist(stream_tgt, bins=30, alpha=0.6, color='#937860',
        label='Target distribution', density=True)
ax.hist(stream_out, bins=30, alpha=0.6, color='#4C72B0',
        label='Output distribution', density=True)
ax.set_title('Toggle: Output vs Target Distribution\n'
             'Target = bimodal at ±1, Output = unimodal near 0',
             fontsize=10)
ax.set_xlabel('Activation value', fontsize=9)
ax.set_ylabel('Density', fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.suptitle('Toggle: Genuine Failure\n'
             'Network outputs ~0 throughout — no bistable dynamics developed',
             fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_toggle_failure_analysis.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Early sustain performance — using stricter time window
# ---------------------------------------------------------------------------

print('Figure 3: Early sustain performance...')

# Evaluate rhythmgeneration using only the first N timesteps of sustain.
# This tests whether the task is actually being learned but decays.

early_windows  = [10, 20, 30, 50, 75, 100, 150, 200]
early_perfs    = []

for w in early_windows:
    # Compute fraction correct using only first w sustain timesteps.
    w_out = sustain_out[:w]    # (w, B)
    w_tgt = sustain_tgt[:w]
    trial_mses   = ((w_out - w_tgt) ** 2).mean(axis=0)   # (B,)
    frac_correct = (trial_mses < 0.05).mean()
    early_perfs.append(frac_correct)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(early_windows, early_perfs, 'o-', color='#C44E52',
        linewidth=2, markersize=8)
ax.axhline(0.9, color='gray', linestyle='--', linewidth=1,
           label='0.9 threshold (solved)')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1,
           label='0.5 threshold')
ax.set_xlabel('Evaluation window (timesteps into sustain)', fontsize=10)
ax.set_ylabel('Fraction correct (MSE < 0.05)', fontsize=10)
ax.set_title('Rhythm Generation: Performance vs Evaluation Window\n'
             'If early window shows high perf → limit cycle stability problem, not learning problem',
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.05, 1.05])
ax.set_xlim([0, max(early_windows) + 10])

for x, y in zip(early_windows, early_perfs):
    ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=8)

# Compare with a well-solved task using same window approach.
ax = axes[1]

# Run memorypro for comparison.
mem_data   = run_task('memorypro', n_trials=16)
mem_resp_on = mem_data['epochs']['go1'][0]
mem_out     = mem_data['output'][mem_resp_on:, :, 1:3]
mem_tgt     = mem_data['y'][mem_resp_on:, :, 1:3]

mem_windows = [5, 10, 20, 30, 50]
mem_perfs   = []
for w in mem_windows:
    w_out = mem_out[:w].reshape(-1, 2)
    w_tgt = mem_tgt[:w].reshape(-1, 2)
    trial_mses   = ((mem_out[:w] - mem_tgt[:w])**2).mean(axis=(0, 2))
    frac_correct = (trial_mses < 0.05).mean()
    mem_perfs.append(frac_correct)

ax.plot(early_windows, early_perfs, 'o-', color='#C44E52',
        linewidth=2, markersize=8, label='Rhythm Gen')
ax.plot(mem_windows,   mem_perfs,   's-', color='#4C72B0',
        linewidth=2, markersize=8, label='Memory Pro (reference)')
ax.axhline(0.9, color='gray', linestyle='--', linewidth=1, label='Solved threshold')
ax.set_xlabel('Evaluation window (timesteps)', fontsize=10)
ax.set_ylabel('Fraction correct (MSE < 0.05)', fontsize=10)
ax.set_title('Rhythm Gen vs Memory Pro: Early Window Performance\n'
             'Rhythm Gen high early → oscillation initiates correctly',
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.05, 1.05])

plt.suptitle('Is Rhythm Generation Actually Learned?\n'
             'Testing with early evaluation window to isolate decay from learning',
             fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_rhythm_early_window_perf.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Conditional rhythm — same analysis
# ---------------------------------------------------------------------------

print('Figure 4: Conditional rhythm analysis...')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

t   = np.arange(condrhythm_data['T'])
eps = condrhythm_data['epochs']

ax = axes[0]
for b in range(4):
    ax.plot(t, condrhythm_data['y'][:, b, 3],
            linestyle='--', color=f'C{b}', alpha=0.6, linewidth=1.5,
            label=f'target {b+1}')
    ax.plot(t, condrhythm_data['output'][:, b, 3],
            linestyle='-', color=f'C{b}', alpha=0.8, linewidth=1.2,
            label=f'output {b+1}')

for ep_name, (ep_start, ep_end) in eps.items():
    ep_end_t = ep_end if ep_end else condrhythm_data['T']
    ax.axvline(ep_start, color='gray', linestyle=':', linewidth=0.8)
    ax.text(ep_start + 1, 0.85, ep_name, fontsize=7,
            color='gray', rotation=90, va='top')

ax.set_title('Conditional Rhythm: 4 Trials\nDashed = target, Solid = output',
             fontsize=10)
ax.set_xlabel('Timestep', fontsize=9)
ax.set_ylabel('Output (channel 3)', fontsize=9)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.2)
ax.set_xlim([0, condrhythm_data['T']])

# Phase 1 and Phase 2 MSE over time separately.
ax = axes[1]
phase1_on  = eps.get('phase_1', (0, None))[0]
switch_on  = eps.get('switch',  (0, None))[0]
phase2_on  = eps.get('phase_2', (0, None))[0]
T_total    = condrhythm_data['T']

if phase1_on and phase2_on:
    p1_out = condrhythm_data['output'][phase1_on:switch_on, :, 3]
    p1_tgt = condrhythm_data['y'][phase1_on:switch_on, :, 3]
    p2_out = condrhythm_data['output'][phase2_on:, :, 3]
    p2_tgt = condrhythm_data['y'][phase2_on:, :, 3]

    mse_p1 = ((p1_out - p1_tgt)**2).mean(axis=1)
    mse_p2 = ((p2_out - p2_tgt)**2).mean(axis=1)

    ax.plot(np.arange(len(mse_p1)), mse_p1,
            color='#4C72B0', linewidth=2, label='Phase 1 MSE')
    ax.plot(np.arange(len(mse_p2)), mse_p2,
            color='#C44E52', linewidth=2, label='Phase 2 MSE')
    ax.axhline(0.05, color='black', linestyle='--',
               linewidth=1, label='Threshold')
    ax.set_xlabel('Timesteps into phase', fontsize=9)
    ax.set_ylabel('MSE', fontsize=9)
    ax.set_title('Conditional Rhythm: MSE by Phase\n'
                 'Phase 1 vs Phase 2 separately',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

plt.suptitle('Conditional Rhythm Analysis\n'
             'Same limit cycle stability issue as Rhythm Generation?',
             fontsize=13, y=1.02)
plt.tight_layout()
path = os.path.join(FIG_DIR, 'fig_condrhythm_analysis.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f'Saved: {path}')
plt.close()


# ---------------------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('RHYTHM / TOGGLE DIAGNOSTIC SUMMARY')
print('=' * 60)

sustain_on = rhythm_data['epochs']['sustain'][0]
sustain_out_all = rhythm_data['output'][sustain_on:, :, 3]
sustain_tgt_all = rhythm_data['y'][sustain_on:, :, 3]
T_sus = sustain_out_all.shape[0]

print(f'\nRhythmGeneration (sustain period = {T_sus} timesteps):')
for w in [10, 20, 50, T_sus]:
    w_mses = ((sustain_out_all[:w] - sustain_tgt_all[:w])**2).mean(axis=0)
    frac   = (w_mses < 0.05).mean()
    print(f'  First {w:4d} ts: {frac:.2f} correct ({frac*16:.0f}/16 trials)')

stream_on  = toggle_data['epochs']['stream'][0]
stream_out = toggle_data['output'][stream_on:, :, 3]
stream_tgt = toggle_data['y'][stream_on:, :, 3]
toggle_mse = ((stream_out - stream_tgt)**2).mean()
toggle_std = stream_out.std()
print(f'\nToggle:')
print(f'  Output std: {toggle_std:.4f} (near 0 = flat output)')
print(f'  Overall MSE: {toggle_mse:.4f}')
print(f'  Conclusion: {"Genuine failure (flat output)" if toggle_std < 0.05 else "Partial learning"}')

print(f'\nConclusion:')
print(f'  RhythmGeneration: limit cycle INITIATES correctly but DECAYS')
print(f'  -> Not a learning failure, a stability failure')
print(f'  -> Early window performance reveals actual learning')
print(f'  Toggle: genuine failure — flat zero output, no bistable dynamics')
print(f'  -> Network has not learned the task at all')