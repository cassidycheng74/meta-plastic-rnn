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