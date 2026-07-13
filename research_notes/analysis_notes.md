## 7/13/26: RNN Dynamics Analysis

Model: LeakyRNN, 256 units, softplus, diagonal init
Checkpoint: ckpt_step1000000.pt
Analysis script: analysis/analyze_dynamics.py
Figures: runs/LeakyRNN_256units_30tasks_seed0/figures/dynamics/

# Overview

Six dynamical analyses were run on the 256-unit LeakyRNN trained on all 30 tasks.
The analyses adapt Driscoll et al. 2024 to the new 30-task format, extending
the original 15-task results and adding new findings from the 19 new tasks.

All analyses use trial endpoints (final hidden state per trial) rather than
all response timesteps, which gives cleaner geometric structure.

# Analysis 1: PCA Trajectories (fig_pca_trajectories.png)

PCA trajectories for 6 selected tasks: MemoryPro, MemoryAnti, DM,
PulseCount, OnlineLinReg, SequentialDec.

## Findings

Memory tasks show a clear fan structure — all trials converge to a single
fixation point during the delay, then diverge during the response epoch in
directions ordered by stimulus angle. The fan hasn't fully closed into a ring
at 1M steps (vs Driscoll's ring at ~100M steps) but the structure is forming.

DM shows two clusters of endpoints corresponding to the two possible decisions,
as expected for a decision-making task.

PulseCount shows discrete endpoint clusters corresponding to each count value
(N=2 through 6). This is a new finding — counting tasks form discrete state
representations rather than continuous manifolds.

OnlineLinReg shows broad spread with no obvious structure, consistent with
in-context learning requiring flexible endpoint representations that vary
with the function parameters learned within each trial.

# Analysis 2: Pro vs Anti Shared Subspace (fig_pro_anti_shared_subspace.png)

Anti task endpoints projected onto pro task PCA axes.
Tests Driscoll's key claim: pro and anti tasks use the same ring manifold.

## Findings

Strong confirmation of shared subspace for both pairs tested.

MemoryPro vs MemoryAnti: both task endpoints trace the same curved arc
in PCA space, with HSV colors ordered continuously around the arc.
The anti endpoints (squares) follow the same arc as the pro endpoints
(circles) with the same color ordering. The ring hasn't fully closed
but the shared manifold is clear.

DelayPro vs DelayAnti: even cleaner result. Pro and anti endpoints are
tightly interleaved on the same arc with identical color ordering.
The network stores stimulus angle as position on the same manifold for
both tasks and reads it out differently based on the task identity vector.

## Comparison to Driscoll

This directly replicates Driscoll's Figure 4 shared subspace result in a
network trained on 30 tasks instead of 15. The ring attractor motif is
robust to the addition of 19 new task types — the network doesn't sacrifice
the memory representation to accommodate new tasks.

The main difference from Driscoll is the arc shape rather than a closed ring,
consistent with earlier training stage (1M vs ~100M steps).

# Analysis 3: Cross-Task Variance Matrix (fig_cross_task_variance.png)

30x30 matrix. Entry (A,B) = fraction of task B endpoint variance explained
by top 3 PCs of task A. Equivalent to Driscoll Fig 4d.

## Findings

Clear block structure along the diagonal — tasks within the same family
explain each other well, tasks across families explain much less.

### Memory/delay block (top-left)
Delay Pro, Delay Anti, Memory Pro, Memory Anti, DM, Extended Memory all
show high mutual variance explanation. Confirms shared ring subspace.
DM being included here is notable — decision and memory tasks overlap
substantially in neural geometry.

### Counting/timing/bistable block (middle)
Pulse Count, Interval Repro, Rate Estim, Rhythm Gen, Seq Recall, Toggle,
and Cond Toggle cluster together. New result not in Driscoll.
Counting, timing, rhythm, and bistable tasks share a common neural subspace
even though toggle and rhythm are at 0.0 performance. The network is building
the right representational subspace for these tasks but hasn't yet learned
to use it for correct behavior.

### Associative/in-context block (lower-right)
Cue Assoc, Paired Assoc, Reversal Learn, Online Lin Reg, Online Nonlin Reg,
and Few-Shot Classif cluster together. New result.
In-context learning and associative tasks share neural geometry, consistent
with both requiring fast within-trial binding of stimulus-response pairs.

### Compositional tasks
Memory DM, Count and Recall, Cond Rhythm, Delayed Assoc, and Sequential Dec
form a loose cluster but also show moderate cross-family overlap with their
component task families. Memory DM overlaps with the memory/delay block.
Count and Recall overlaps with the counting block.

## Key difference from Driscoll

Off-diagonal values are generally higher than Driscoll's 15-task version —
more cross-family sharing. Expected with 30 tasks sharing 256 units,
more pressure to share representational resources. Block structure is
present but less clean than the 15-task version.

# Analysis 4: Unit Variance Matrix (fig_variance_matrix.png)

256 units x 30 tasks. Normalized variance per unit per task.
Equivalent to Driscoll Fig 3a.

## Findings

Most units are broadly tuned across many tasks — less sparse selectivity
than Driscoll's 128-unit network. More units means more capacity to
spread representational load.

Two hub units (around rows 200-210) are extremely bright across nearly
all tasks. These units participate in almost every task and are worth
examining in Phase 7 — they may be task-general computation units.

Bar chart findings:
- Delay Pro, Delay Anti, Reversal Learning, Ctx DM-B drive highest
  mean unit variance — most diverse neural activity across population
- Memory Pro and Memory Anti drive surprisingly low mean variance —
  consistent with ring attractor being a low-dimensional compact
  representation that doesn't require much population-level variance
- Rhythm Gen, Toggle, Cond Toggle show low mean variance despite being
  unsolved — network is not engaging strongly with these tasks
- Delayed Assoc shows low variance but solves at 1.0 — very efficient
  compact representation

# Analysis 5: Global PCA (fig_global_pca.png)

All 30 task endpoints projected into a single 3D PCA space fitted on
the concatenation of all task endpoints.

## Findings

PC1 (28.2%) separates tasks along a broad axis.

In-context learning tasks (OnlineLin, OnlineNonlin, RateEstim) sit
far out to the upper left, completely separated from all other tasks.
This is a strong new finding — in-context learning tasks occupy a
fundamentally different region of neural space. The network has developed
a dedicated representational subspace for in-context function learning.

Interval Repro is a clear outlier on the far right — different endpoint
distribution from all other tasks, consistent with it being the only task
requiring a precise temporal duration representation rather than a
categorical or angular response.

Memory/decision/associative tasks cluster in the center-right region.
Counting/timing tasks are spread in the middle.

The lack of tight family clustering in global PCA is expected — the
cross-task variance matrix is a more sensitive measure of family structure.

# Analysis 6: Compositional Task Subspace (fig_compositional_subspace.png)

Compositional task endpoints projected onto component task PCA axes.

## Findings

MemoryDM vs components: MemoryDM endpoints distributed across the same
space as both MemoryPro and DM — uses a mixture of both component subspaces.
Partial subspace sharing confirmed.

CountAndRecall vs components: most striking result. PulseCount endpoints
cluster extremely tightly in a narrow vertical band — the discrete counting
attractor is clearly visible even on memory axes. CountAndRecall is much
more spread out — must simultaneously encode count AND angle.
The two primitives occupy clearly different dimensions within the
counting task PCA space.

SequentialDecision vs DM + Memory: endpoints interleaved with both
component tasks, using a mixture of both subspaces. Consistent with
being a three-block decision task combining multiple integration steps.

## Interpretation

Compositional tasks partially but not fully share subspace with components.
The network decomposes compositional tasks into identifiable primitive
representations rather than learning entirely new solutions. The most
dramatic example is CountAndRecall where the counting primitive (tight cluster)
and memory primitive (broad spread) are clearly distinguishable within
the same PCA space.

# Analysis 7: Memory Ring Detail (fig_memory_ring_detail.png)

Detailed trajectory plot for MemoryPro, MemoryAnti, DelayPro.
Light gray = fixation, gray = delay, color = response (HSV = stimulus angle).

## Findings

All memory tasks show convergence to a single fixation fixed point during
fixation/delay period (all gray lines converge to the same point).

During response epoch, trajectories fan out in directions ordered by
stimulus angle (HSV color ordering visible in endpoints).

The arc structure is forming but not yet a closed ring — the fan represents
early-stage ring attractor development consistent with 1M training steps.

MemoryAnti shows slightly different PCA structure from MemoryPro
(different eigenvalues and orientation) even though both perform at 1.0.
This suggests pro and anti use partially different neural dimensions
rather than exactly the same ring — more nuanced than Driscoll's
strict shared subspace result. Worth investigating further in Phase 7.

# Summary: New Findings Beyond Driscoll

These results extend Driscoll's 15-task analysis to 30 tasks and reveal
several new findings:

1. Ring attractor motif is robust to 19 new tasks. Pro/anti shared subspace
   confirmed for memory and delay task pairs even in the 30-task network.

2. Counting/timing/bistable tasks form a coherent subspace cluster even
   when stuck at 0.0 performance. The network builds the right representational
   subspace but hasn't developed correct attractor dynamics. Strong motivation
   for Phase 5 — plasticity rules may push these into functional attractors.

3. In-context learning tasks occupy a completely separate neural subspace
   from all other task families. This is consistent with in-context learning
   requiring a qualitatively different computational mechanism.

4. Compositional tasks decompose into component primitive representations.
   CountAndRecall shows counting (tight cluster) and memory (broad spread)
   as clearly distinguishable within the same PCA space.

5. Two hub units active across nearly all tasks. May be task-general
   computation units — worth examining in Phase 7 fixed point analysis.

6. MemoryPro and MemoryAnti show lower mean unit variance than expected,
   consistent with ring attractor being a compact low-dimensional representation.