## Phase 2/3: 30-Task Baseline Training
Date: 2026-06-29

### What was built
- New repo: meta-plastic-rnn (PyTorch, clean 11-input/5-output format)
- tasks/base.py: Trial class, TaskDataset, channel constants
- tasks/yang_driscoll.py: 11 Yang/Driscoll tasks ported to new format
- tasks/new_tasks.py: 19 new tasks (T12-T30)
- network/rnn.py: LeakyRNN and GRU in PyTorch
- network/train.py: Training loop with curriculum support
- scripts/test_tasks.py: Sanity check — all 30 tasks pass
- scripts/train_all30.py: Full 30-task training script
- scripts/submit_slurm.sh: SLURM batch job

### Training job submitted
- Job: LeakyRNN, 128 units, seed 0, max_steps=10M
- Curriculum: Yang/Driscoll 11 tasks for 500k steps, then all 30
- Account: kempner_dgc_lab, partition: kempner, 1xA100, 12h
- Output: runs/LeakyRNN_128units_30tasks_seed0/

### Key differences from Driscoll repo
- PyTorch instead of TF1 — easier to modify for plasticity work
- 11-input/5-output unified format instead of ring encoding
- 30 tasks instead of 15 — adds counting, timing, rhythm, associative tasks
- Curriculum training — Yang/Driscoll warmup before new tasks
- Explicit per-timestep loop in LeakyRNN.forward() — required for Phase 5

### What to look for when training finishes
- Yang/Driscoll tasks should match Driscoll reproduction performance
- New tasks should show meaningful learning above chance
- T20/T22 will learn supervised proxy versions only
- T16/T18/T28 (rhythm, toggle) may need more steps — limit cycle dynamics

# 07-06-26


---

## Run: 30-Task LeakyRNN Baseline (128 units, v3)

**Date:** 2026-07-07
**Status:** Complete (hit time limit)
**Log:** `runs/LeakyRNN_128units_30tasks_seed0_v3/`

---

### Setup

| Parameter | Value |
|-----------|-------|
| Architecture | LeakyRNN |
| Units | 128 |
| Activation | softplus |
| Weight init | diagonal |
| Tasks | 30 (11 Yang/Driscoll + 19 new) |
| Max steps | 10,000,000 |
| Steps completed | 1,110,000 |
| Learning rate | 3e-4 |
| Batch size | 128 |
| Display step | 5,000 |
| Checkpoint step | 50,000 |
| Task identity fix | ❌ Not yet applied |
| GPU | A100-SXM4-40GB |
| Wall time | ~48 hours |

---

### Training Dynamics

- Loss: high variance throughout (0.01–0.2), expected given random task sampling
- Loss envelope flat after ~200k steps → network hit capacity ceiling early
- `perf_avg` plateaued at ~0.70–0.75 after ~100k steps, no further improvement
- `perf_min` stayed at 0.0 entire run (stuck tasks dragging minimum down)
- Training speed: stable ~6.45 steps/second on GPU

---

### Per-Family Results

#### ✅ Solved and Stable

| Family | Tasks | Notes |
|--------|-------|-------|
| Counting/Timing | pulsecounting, intervalreproduction, pulserateestimation | Hit 1.0 by ~50k steps, perfectly stable |
| In-Context Learning | onlinelinearreg, onlinenonlinearreg, fewshotclassif | Hit 1.0 almost immediately — faster than Counting |
| Associative | cueresponseassoc, pairedassociation, reversallearning, multiitemrecall | Converged ~200k steps, stable near 1.0 |
| Compositional | memorydm, countandrecall, delayedassociation, sequentialdecision | Converged ~200k steps, stable 0.95–1.0 |

#### 🔄 Partially Learned / Unstable

| Family | Tasks | Notes |
|--------|-------|-------|
| Decision | dm, dmanti, contextdm_a, contextdm_b | 0.7–0.95, stable |
| Match | delaymatchsample | Near 1.0, stable |
| Match | delaynonmatchsample | Oscillating 0.3–1.0 — task identity issue |
| Memory | delaypro, memorypro | Oscillating 0–1.0 wildly — task identity issue |
| Memory | extendedmemory | Near 1.0, stable |
| Rhythm | sequencerecall | 0.4–0.8, noisy |
| Rhythm | rhythmgeneration | Intermittent, never converged |

#### ❌ Never Learned (0.0 entire run)

| Family | Tasks | Likely cause |
|--------|-------|--------------|
| Memory | delayanti, memoryanti | Task identity bug |
| Rhythm | conditionalrhythm | Limit cycle dynamics |
| Bistable | toggle, conditionaltoggle | Bistable dynamics — needs more capacity/steps |

---

### Final Performance at Step 1,110,000

| Task | Perf | Task | Perf |
|------|------|------|------|
| delaypro | ~0.22 (unstable) | pulsecounting | 1.00 |
| delayanti | 0.00 | intervalreproduction | 1.00 |
| memorypro | ~0.50 (unstable) | pulserateestimation | 1.00 |
| memoryanti | 0.00 | rhythmgeneration | ~0.50 (unstable) |
| extendedmemory | 1.00 | sequencerecall | 0.25 |
| dm | 0.94 | conditionalrhythm | 0.00 |
| dmanti | 0.78 | toggle | 0.00 |
| contextdm_a | 0.88 | conditionaltoggle | 0.00 |
| contextdm_b | 0.84 | cueresponseassoc | 1.00 |
| delaymatchsample | 1.00 | pairedassociation | 0.91 |
| delaynonmatchsample | 0.50 | reversallearning | 1.00 |
| onlinelinearreg | 1.00 | multiitemrecall | 0.97 |
| onlinenonlinearreg | 1.00 | memorydm | 1.00 |
| fewshotclassif | 1.00 | countandrecall | 1.00 |
| sequentialdecision | 0.91 | delayedassociation | 1.00 |

---

### Key Bug Identified: Missing Task Identity Signal

**Problem:** Tasks with identical input structures but different required outputs cannot
be learned simultaneously. The network receives contradictory gradients and defaults
to whichever mapping it learned first.

Affected task pairs:
- `delaypro` vs `delayanti` — identical inputs, opposite responses
- `memorypro` vs `memoryanti` — identical inputs, opposite responses
- `delaymatchsample` vs `delaynonmatchsample` — identical inputs, inverted convention

**Fix:** `TaskDataset` in `tasks/base.py` now injects a unique random ±1 vector into
input channels 7–10 for every trial of every task. This gives the network a stable
signal to distinguish tasks with identical sensory inputs. Equivalent to Driscoll's
rule input vector. Applied in next run.

---

### Scientific Notes

1. **In-context learning tasks solved immediately** — onlinelinearreg, onlinenonlinearreg,
   fewshotclassif all hit 1.0 within the first few thousand steps. A fixed-weight RNN
   solving in-context function learning this easily is unexpected. Worth investigating
   whether the network is genuinely learning to fit functions from examples or exploiting
   a shortcut in the task design (e.g. the linear structure makes gradient descent trivial).

2. **Compositional tasks as easy as simpler tasks** — memorydm, countandrecall,
   delayedassociation all converged as fast as the component tasks they combine. Suggests
   the network transfers learned primitives effectively. Positive sign for the
   meta-learning phase.

3. **Bistable/rhythm tasks need more than capacity** — toggle and rhythm tasks showed
   zero learning even though other hard tasks (compositional, associative) converged
   easily. These require qualitatively different dynamics (limit cycles, bistable switches)
   not just more parameters. May need specialized initialization or longer training.

---

### Changes for Next Run

**Run:** `runs/LeakyRNN_256units_30tasks_seed0`

| Change | v3 (this run) | v4 (next run) | Reason |
|--------|--------------|---------------|--------|
| Units | 128 | 256 | PI recommendation, more capacity |
| Task identity | ❌ | ✅ | Fix delayanti/memoryanti bug |
| Batch size | 128 | 128 | No change |
| Learning rate | 3e-4 | 3e-4 | No change |
| Display step | 5,000 | 5,000 | No change |
| Time limit | 2 days | 2 days | Cluster max |

**Expected improvements:**
- delayanti and memoryanti should learn with task identity fix
- DNMS instability should resolve
- Higher performance ceiling overall with 256 units
- Rhythm/bistable tasks still expected to struggle

---

### Figures

All figures saved to `runs/LeakyRNN_128units_30tasks_seed0_v3/figures/`:

- `fig_training_summary.png` — loss, avg/min perf, training speed over time
- `fig_pertask_by_family.png` — per-task learning curves grouped by family ← key figure
- `fig_all30_performance.png` — all 30 tasks on one plot colored by family
- `fig_final_performance_bar.png` — bar chart of final performance by family

---

### Next Steps

- [ ] Wait for 256-unit run to finish (~2 days)
- [ ] Run `plot_training_curves_30tasks.py` on new run and compare to these figures
- [ ] Add per-task performance logging to `train.py` so future analysis doesn't
      require reloading checkpoints
- [ ] Investigate why in-context learning tasks solve so easily
- [ ] Read Miconi differentiable plasticity paper for Phase 5 prep
- [ ] Meet with PI to confirm Phase 3 is sufficient to move to Phase 5

# 07-10-26
## 256-unit RNN Run
Architecture: GRU
Units: 256
Tasks: 30 (11 Yang/Driscoll + 19 new)
Max steps: 10,000,000
Learning rate: 3e-4
Batch size: 128
Display step: 5,000
Checkpoint step: 50,000
Task identity: yes (base.py + train_v3.py)
task_vecs saved in checkpoint: yes
Per-task perf in log: yes
GPU: A100-SXM4-40GB
Time limit: 2 days

## Final performance at step 1,000,000
Evaluated with correct task_vec injection, 5 batches per task.
26/30 tasks solved.
Solved at 1.0: delaypro, delayanti, memorypro, memoryanti, extendedmemory, contextdm_a, contextdm_b, delaymatchsample, delaynonmatchsample, pulsecounting, intervalreproduction, pulserateestimation, sequencerecall, cueresponseassoc, pairedassociation, reversallearning, onlinelinearreg, onlinenonlinearreg, fewshotclassif, countandrecall, delayedassociation.
Near 1.0: dm (0.975), dmanti (0.963), multiitemrecall (0.975), memorydm (0.938), sequentialdecision (0.988).
Stuck at 0.0: rhythmgeneration, conditionalrhythm, toggle, conditionaltoggle.

## Key findings
- Task identity vectors essential. delayanti and memoryanti were 0.0 without them, 1.0 with them. - Equivalent to Driscoll's rule input.
- 256 units sufficient for 26/30 tasks. The 4 stuck tasks need qualitatively different dynamics, not more capacity.
- In-context learning solved immediately. onlinelinearreg and fewshotclassif at 1.0 within first few thousand steps.
- Compositional tasks as easy as components. Good primitive transfer. Positive sign for meta-learning phase.
- Loss reached 0.0038, still decreasing at job cutoff
- More training would likely push dm and memorydm to 1.0.

## 7/13/26

# Model Performance Comparison

=================================================================
FINAL PERFORMANCE COMPARISON
=================================================================
Task                        LeakyRNN        GRU  Transformer
-----------------------------------------------------------------

Yang/Driscoll Memory
  Delay Pro                    1.000      1.000        1.000
  Delay Anti                   1.000      1.000        1.000
  Memory Pro                   1.000      1.000        1.000
  Memory Anti                  1.000      1.000        1.000
  Ext Memory                   1.000      1.000        1.000

Yang/Driscoll Decision
  DM                           0.979      1.000        0.938
  DM Anti                      1.000      1.000        0.979
  Ctx DM-A                     1.000      1.000        1.000
  Ctx DM-B                     1.000      1.000        1.000

Yang/Driscoll Match
  DMS                          1.000      1.000        1.000
  DNMS                         1.000      1.000        1.000

Counting/Timing
  Pulse Count                  1.000      1.000        1.000
  Interval Repro               1.000      1.000        1.000
  Rate Estim                   1.000      1.000        1.000

Rhythm/Sequence
  Rhythm Gen                   0.000      0.000        0.000
  Seq Recall                   1.000      1.000        1.000
  Cond Rhythm                  0.000      0.021        0.000

Bistable
  Toggle                       0.000      0.000        0.000
  Cond Toggle                  0.000      0.000        0.000

Associative
  Cue Assoc                    1.000      1.000        1.000
  Paired Assoc                 1.000      1.000        1.000
  Reversal Learn               1.000      1.000        1.000
  Multi-Item Recall            1.000      0.958        0.896

In-Context Learning
  Online Lin Reg               1.000      1.000        1.000
  Online Nonlin Reg            1.000      1.000        1.000
  Few-Shot Classif             1.000      1.000        1.000

Compositional
  Memory DM                    0.979      0.979        1.000
  Count & Recall               1.000      1.000        0.979
  Delayed Assoc                1.000      1.000        1.000
  Sequential Dec               0.979      0.979        1.000

-----------------------------------------------------------------
  Tasks solved (>=0.9)            26         26           25

  Run: 30-Task Transformer Baseline (d_model=128)

# Transformer Run
d_model: 128
n_heads: 4
n_layers: 3
d_ff: 256
Tasks: 30
Steps completed: 5,700,000
Learning rate: 1e-4
Batch size: 64
Task identity: yes
task_vecs saved: yes
Per-task perf in log: yes
GPU: A100-SXM4-40GB

# Final performance at 5.7M steps

26/30 tasks solved — identical profile to LeakyRNN and GRU.

Solved at 1.0: delaypro, delayanti, memorypro, memoryanti, extendedmemory, contextdm_a, contextdm_b, delaymatchsample, delaynonmatchsample, pulsecounting, intervalreproduction, pulserateestimation, sequencerecall, cueresponseassoc, pairedassociation, reversallearning, onlinelinearreg, onlinenonlinearreg, fewshotclassif, countandrecall, delayedassociation.

Near 1.0: dm (0.975), dmanti (0.988), multiitemrecall (0.988), memorydm (0.938), sequentialdecision (0.988).

Stuck at 0.0: rhythmgeneration, conditionalrhythm, toggle, conditionaltoggle.

# Notes

Warning about enable_nested_tensor is harmless — caused by norm_first=True in TransformerEncoderLayer. Does not affect results.

The transformer solved all the same tasks as the RNN with identical final performance numbers. The scientific interest is in the mechanism, not the performance — Phase 7 will compare attention patterns vs ring attractor dynamics for memory tasks.

# GRU Run

Architecture: GRU
Units: 256
Tasks: 30
Steps completed: 4,300,000
Learning rate: 3e-4
Batch size: 128
Task identity: yes
task_vecs saved: yes
Per-task perf in log: yes
GPU: A100-SXM4-40GB

# Final performance at 4.3M steps

26/30 tasks solved.

Solved at 1.0: delaypro, delayanti, memorypro, memoryanti, extendedmemory, contextdm_a, contextdm_b, delaymatchsample, delaynonmatchsample, pulsecounting, intervalreproduction, pulserateestimation, sequencerecall, cueresponseassoc, pairedassociation, reversallearning, onlinelinearreg, onlinenonlinearreg, fewshotclassif, countandrecall, delayedassociation.

Near 1.0: dm (0.975), dmanti (0.988), multiitemrecall (0.988), memorydm (0.938), sequentialdecision (0.988).

Stuck at 0.0: toggle, conditionaltoggle, conditionalrhythm.

Rhythmgeneration: 0.013 — first nonzero value on this task across any architecture. Small but notable.

# Key findings

All three architectures converge to the same performance profile — 26/30 tasks solved, same 4 stuck at 0.0. This is the main Phase 3 result.

The stuck tasks (toggle, conditionaltoggle, rhythmgeneration, conditionalrhythm) are confirmed architecture-independent failures. They require bistable and limit cycle dynamics that fixed-weight networks of any type cannot reliably develop within this training budget. This motivates Phase 5 — meta-learned plasticity rules may scaffold these dynamics.

The only difference across architectures is that GRU shows 0.013 on rhythmgeneration — the first nonzero value on this task. Suggestive but not conclusive.

The scientific interest in Phase 7 is the mechanism, not the performance. All three architectures solve the same tasks but almost certainly via different internal computations — ring attractors (LeakyRNN), gated dynamics (GRU), attention patterns (Transformer). PCA and shared subspace analysis will reveal whether different architectures find the same or different dynamical motifs for the same tasks.

## 7/30/26

Run: LeakyRNN 256 units — All 30 Tasks v4 (Fixed Task Definitions)

Date: 2026-07-28
Status: Complete, hit 2-day time limit at 1M steps
Log: runs/LeakyRNN_256units_30tasks_seed0_v4/

Architecture: LeakyRNN, 256 units
Steps completed: 1,000,000
Task identity: yes
Task fixes applied: yes (toggle initial state, rhythm ramp, conditionalrhythm phase continuity)

Final performance at 1M steps

26/30 tasks solved — identical profile to v1. Task fixes did not break any previously solved tasks.

Toggle and rhythm still at 0.0 in multitask setting — multitask interference confirmed as a factor. The isolated runs (below) show these tasks can learn when trained alone.

Notable: dm (0.975), dmanti (0.963), memorydm (0.938), multiitemrecall (0.975), countandrecall (0.988), sequentialdecision (0.988) all near but not at 1.0 — would likely converge with more steps.

Run: LeakyRNN 256 units — Rhythm Only (Fixed Tasks)

Date: 2026-07-28
Status: Complete, hit 2-day time limit at 650k steps
Log: runs/LeakyRNN_256units_rhythm_fixed_seed0/

Architecture: LeakyRNN, 256 units
Tasks: rhythmgeneration, conditionalrhythm only
Task fixes applied: yes (ramp-up, conditionalrhythm phase continuity)

Final performance at 650k steps

rhythmgeneration: 0.062
conditionalrhythm: 0.000

First nonzero performance ever seen on rhythmgeneration across any architecture or training run. Performance is low and noisy (0.03-0.16 during training) but confirms the task is learnable in principle with the right dynamics.

ConditionalRhythm still at 0.0 — harder variant, needs rhythmgeneration to stabilize first.

Key finding

The ramp-up fix (5-timestep amplitude ramp at sustain start) was necessary but not sufficient. The network can occasionally initiate the correct oscillation but cannot sustain it reliably — limit cycle stability is still the bottleneck. More steps and/or plasticity rules needed.

Run: LeakyRNN 256 units — Toggle Only (Fixed Tasks)

Date: 2026-07-28
Status: Complete, hit 2-day time limit at 1.05M steps
Log: runs/LeakyRNN_256units_toggle_fixed_seed0/

Architecture: LeakyRNN, 256 units
Tasks: toggle, conditionaltoggle only
Task fixes applied: yes (initial state signaled on IN_REAL_B)

Final performance at 1.05M steps

toggle: 0.050
conditionaltoggle: 0.000

First nonzero performance ever seen on toggle. Performance fluctuates 0.00-0.12 during training, settling around 0.05.

ConditionalToggle still at 0.0 — requires two independent bistable states simultaneously, needs toggle to stabilize first.

Key finding

The initial state signal fix was necessary and sufficient to get nonzero performance. Previously the network had no information about the starting state and averaged to zero. With the fix, the network can occasionally initialize correctly and track a few flips, but bistable attractor dynamics are still fragile.

Task Bug Analysis and Fixes (2026-07-27)

Four bugs identified and fixed in tasks/new_tasks.py:

T16 RhythmGeneration: target amplitude now ramps from 0 to 1 over first 5 timesteps of sustain. Previously required instantaneous phase initialization from arbitrary hidden state.

T18 Toggle: initial bistable state now signaled on IN_REAL_B for first 10 timesteps of stream. Previously network had no way to know starting state and averaged to zero output.

T19 ConditionalToggle: initial states for both channels now signaled (channel A on IN_REAL_B, channel B on IN_SIN_B). Same fix as toggle applied to both independent channels.

T28 ConditionalRhythm: Phase 2 target now maintains phase continuity across the switch. Previously reset to phase=0 creating a discontinuity that made smooth tracking physically impossible.

Impact: toggle and rhythmgeneration show nonzero performance for the first time after fixes. ConditionalToggle and ConditionalRhythm still at 0.0 — harder variants need simpler versions to converge first.