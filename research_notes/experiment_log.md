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