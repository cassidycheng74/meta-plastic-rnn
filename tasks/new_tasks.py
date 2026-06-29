"""
tasks/new_tasks.py

New tasks T12-T30 for the meta-plastic-rnn project.
All use the unified 11-input / 5-output Trial format from tasks/base.py.

Task list:
    T12  MultiItemRecall       - multi-item memory + associative binding
    T13  PulseCounting         - counting
    T14  IntervalReproduction  - interval timing (Ready-Set-Go)
    T15  PulseRateEstimation   - rate estimation
    T16  RhythmGeneration      - limit cycle / frequency cue
    T17  SequenceRecall        - sequence memory + generation
    T18  Toggle                - bistable flip-flop
    T19  ConditionalToggle     - two independent flip-flops
    T20  CueResponseAssoc      - cue->angle mapping (supervised proxy)
    T21  PairedAssociation     - within-trial one-shot binding
    T22  ReversalLearning      - cue->angle with reversal signal
    T23  OnlineLinearReg       - in-context linear regression
    T24  OnlineNonlinearReg    - in-context nonlinear regression
    T25  FewShotClassif        - K-shot classification
    T26  MemoryDM              - memory + decision making (compositional)
    T27  CountAndRecall        - counting + binding (compositional)
    T28  ConditionalRhythm     - rhythm with mid-trial frequency switch
    T29  DelayedAssociation    - binding + interference resistance
    T30  SequentialDecision    - three sequential decisions (compositional)

Each function signature:
    task_fn(config: dict, rng: np.random.RandomState) -> Trial

Input channels (from tasks/base.py):
    IN_FIX    = 0   fixation
    IN_SIN_A  = 1   sin of angle A
    IN_COS_A  = 2   cos of angle A
    IN_SIN_B  = 3   sin of angle B
    IN_COS_B  = 4   cos of angle B
    IN_REAL_A = 5   scalar input A  (pulses, rates, scalars)
    IN_REAL_B = 6   scalar input B
    IN_CUE_0..3 = 7-10  4-dim cue vector

Output channels:
    OUT_FIX    = 0   fixation
    OUT_SIN    = 1   sin of response angle  (also used as scalar A output)
    OUT_COS    = 2   cos of response angle  (also used as scalar B output)
    OUT_REAL_A = 3   scalar response A
    OUT_REAL_B = 4   scalar response B
"""

from __future__ import annotations

import numpy as np
from tasks.base import (
    Trial,
    IN_FIX, IN_SIN_A, IN_COS_A, IN_SIN_B, IN_COS_B,
    IN_REAL_A, IN_REAL_B,
    IN_CUE_0, IN_CUE_1, IN_CUE_2, IN_CUE_3,
    OUT_FIX, OUT_SIN, OUT_COS, OUT_REAL_A, OUT_REAL_B,
    N_INPUT, N_OUTPUT,
    random_cue_vectors,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ri(rng, lo, hi):
    """Random int in [lo, hi] inclusive."""
    return int(rng.randint(lo, hi + 1))


def _ru(rng, lo, hi):
    """Random float in [lo, hi)."""
    return float(rng.uniform(lo, hi))


def _fix(trial, t0, t1):
    """Set fixation input and output over [t0, t1)."""
    trial.x[t0:t1, :, IN_FIX]  = 1.0
    trial.y[t0:t1, :, OUT_FIX] = 1.0


def _angle_in(trial, angles, channel, t0, t1):
    """Write sin/cos of angles into input channels over [t0, t1)."""
    if channel == 'A':
        trial.x[t0:t1, :, IN_SIN_A] = np.sin(angles)
        trial.x[t0:t1, :, IN_COS_A] = np.cos(angles)
    else:
        trial.x[t0:t1, :, IN_SIN_B] = np.sin(angles)
        trial.x[t0:t1, :, IN_COS_B] = np.cos(angles)


def _angle_out(trial, angles, t0, t1):
    """Write sin/cos of angles into output channels over [t0, t1)."""
    trial.y[t0:t1, :, OUT_SIN] = np.sin(angles)
    trial.y[t0:t1, :, OUT_COS] = np.cos(angles)


def _cue_in(trial, vectors, t0, t1):
    """
    Write a cue vector into inputs 7-10 over [t0, t1).

    Args:
        vectors: (B, 4) or (4,) array of ±1 values
    """
    trial.x[t0:t1, :, IN_CUE_0:IN_CUE_3 + 1] = vectors


# ---------------------------------------------------------------------------
# T12: MultiItemRecall
# ---------------------------------------------------------------------------

def multiitemrecall(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T12: MultiItemRecall.

    K items shown sequentially, each = angle on channel A + cue vector on
    channels 7-10. After a delay, one cue vector is shown alone as a probe.
    Respond with the angle originally paired with that cue.

    Inputs:  fixation, angle A, cue vector (7-10)
    Outputs: fixation, angle response
    """
    dt = config['dt']
    B  = config['batch_size']
    K  = _ri(rng, 2, 4)

    fix_dur   = _ri(rng, 30, 80)
    item_dur  = _ri(rng, 50, 80)
    delay_dur = _ri(rng, 50, 200)
    probe_dur = _ri(rng, 40, 80)
    resp_dur  = _ri(rng, 30, 100)

    items_dur = K * item_dur
    probe_on  = fix_dur + items_dur + delay_dur
    probe_off = probe_on + probe_dur
    resp_on   = probe_off
    tdim      = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)

    # Generate K (angle, cue_vector) pairs per batch item.
    # Cues are fresh per trial.
    angles     = rng.uniform(0, 2 * np.pi, (B, K))
    probe_idxs = rng.randint(0, K, size=B)

    for k in range(K):
        t0 = fix_dur + k * item_dur
        t1 = t0 + item_dur
        cues_k = random_cue_vectors(B, rng)   # (B, 4)
        for i in range(B):
            trial.x[t0:t1, i, IN_SIN_A] = np.sin(angles[i, k])
            trial.x[t0:t1, i, IN_COS_A] = np.cos(angles[i, k])
            trial.x[t0:t1, i, IN_CUE_0:IN_CUE_3 + 1] = cues_k[i]

        # Store cues for probe lookup.
        if k == 0:
            all_cues = cues_k[:, np.newaxis, :]   # (B, 1, 4)
        else:
            all_cues = np.concatenate(
                [all_cues, cues_k[:, np.newaxis, :]], axis=1)   # (B, K, 4)

    # Probe: show the cue vector for the probed item.
    for i in range(B):
        k_probe = probe_idxs[i]
        trial.x[probe_on:probe_off, i, IN_CUE_0:IN_CUE_3 + 1] = all_cues[i, k_probe]
        trial.y[resp_on:, i, OUT_SIN] = np.sin(angles[i, k_probe])
        trial.y[resp_on:, i, OUT_COS] = np.cos(angles[i, k_probe])

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,              fix_dur),
        'items' : (fix_dur,        fix_dur + items_dur),
        'delay1': (fix_dur + items_dur, probe_on),
        'probe' : (probe_on,       probe_off),
        'go1'   : (resp_on,        None),
    }
    return trial


# ---------------------------------------------------------------------------
# T13: PulseCounting
# ---------------------------------------------------------------------------

def pulsecounting(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T13: PulseCounting.

    N irregular pulses arrive on IN_REAL_A. After a delay, output N/N_max
    as a scalar on OUT_REAL_A. Not solvable by simple integration.

    Inputs:  fixation, pulse train on IN_REAL_A
    Outputs: fixation, scalar count on OUT_REAL_A
    """
    dt    = config['dt']
    B     = config['batch_size']
    N_max = 6

    n_pulses  = rng.randint(2, N_max + 1, size=B)
    fix_dur   = _ri(rng, 30, 80)
    pulse_dur = _ri(rng, 100, 300)
    delay_dur = _ri(rng, 30, 100)
    resp_dur  = _ri(rng, 30, 100)

    pulse_on  = fix_dur
    pulse_off = fix_dur + pulse_dur
    resp_on   = pulse_off + delay_dur
    tdim      = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)

    pulse_w = max(1, int(5 / dt))
    for i in range(B):
        available = np.arange(pulse_on, pulse_off - pulse_w)
        positions = sorted(rng.choice(available, size=n_pulses[i], replace=False))
        for p in positions:
            trial.x[p:min(p + pulse_w, pulse_off), i, IN_REAL_A] = 1.0
        trial.y[resp_on:, i, OUT_REAL_A] = n_pulses[i] / N_max

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,         pulse_on),
        'pulses': (pulse_on,  pulse_off),
        'delay1': (pulse_off, resp_on),
        'go1'   : (resp_on,   None),
    }
    return trial


# ---------------------------------------------------------------------------
# T14: IntervalReproduction
# ---------------------------------------------------------------------------

def intervalreproduction(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T14: IntervalReproduction (Ready-Set-Go).

    Two pulses define interval delta_t. After the second pulse (Set),
    produce a pulse delta_t later. Jazayeri & Shadlen paradigm.

    Inputs:  fixation, pulses on IN_REAL_A
    Outputs: fixation, produced pulse on OUT_REAL_A
    """
    dt      = config['dt']
    B       = config['batch_size']
    pulse_w = max(1, int(5 / dt))

    delta_t  = _ri(rng, 50, 200)
    fix_dur  = _ri(rng, 30, 80)
    prod_dur = _ri(rng, 200, 400)

    ready_on  = fix_dur
    ready_off = ready_on + pulse_w
    set_on    = ready_off + delta_t
    set_off   = set_on + pulse_w
    target_t  = set_off + delta_t
    tdim      = target_t + prod_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    # Fixation stays on throughout — no traditional go cue.
    trial.x[:, :, IN_FIX]  = 1.0
    trial.y[:, :, OUT_FIX] = 1.0

    # Ready pulse.
    trial.x[ready_on:ready_off, :, IN_REAL_A] = 1.0
    # Set pulse.
    trial.x[set_on:set_off, :, IN_REAL_A] = 1.0
    # Target: brief pulse centered at target_t.
    t0 = max(0, target_t - pulse_w)
    t1 = min(tdim, target_t + pulse_w)
    trial.y[t0:t1, :, OUT_REAL_A] = 1.0

    trial.add_cost_mask(response_on=set_off)
    trial.epochs = {
        'fix1'    : (0,        ready_on),
        'ready'   : (ready_on, ready_off),
        'interval': (ready_off, set_on),
        'set'     : (set_on,   set_off),
        'prod'    : (set_off,  None),
    }
    return trial


# ---------------------------------------------------------------------------
# T15: PulseRateEstimation
# ---------------------------------------------------------------------------

def pulserateestimation(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T15: PulseRateEstimation.

    Poisson pulse train at rate r over a fixed window.
    Output r / r_max as scalar after window ends.

    Inputs:  fixation, pulse train on IN_REAL_A
    Outputs: fixation, scalar rate on OUT_REAL_A
    """
    dt    = config['dt']
    B     = config['batch_size']
    r_max = 0.30

    rates    = rng.uniform(0.05, r_max, size=B)
    fix_dur  = _ri(rng, 30, 80)
    obs_dur  = 200   # fixed observation window
    resp_dur = _ri(rng, 30, 100)

    obs_on  = fix_dur
    obs_off = fix_dur + obs_dur
    resp_on = obs_off
    tdim    = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)

    pulse_w = max(1, int(3 / dt))
    for i in range(B):
        for t in range(obs_on, obs_off):
            if rng.rand() < rates[i]:
                trial.x[t:min(t + pulse_w, obs_off), i, IN_REAL_A] = 1.0
        trial.y[resp_on:, i, OUT_REAL_A] = rates[i] / r_max

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'   : (0,       obs_on),
        'observe': (obs_on,  obs_off),
        'go1'    : (resp_on, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T16: RhythmGeneration
# ---------------------------------------------------------------------------

def rhythmgeneration(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T16: RhythmGeneration.

    Brief scalar cue specifies frequency f. Network must sustain
    sin(2*pi*f*t) on OUT_REAL_A. Requires limit cycle dynamics.

    Inputs:  fixation during cue, frequency cue on IN_REAL_A
    Outputs: fixation during cue, sustained sinusoid on OUT_REAL_A
    """
    dt    = config['dt']
    B     = config['batch_size']
    f_min = 0.02
    f_max = 0.08

    freqs       = rng.uniform(f_min, f_max, size=B)
    fix_dur     = _ri(rng, 30, 80)
    cue_dur     = _ri(rng, 20, 40)
    sustain_dur = _ri(rng, 200, 400)

    cue_on  = fix_dur
    cue_off = fix_dur + cue_dur
    tdim    = cue_off + sustain_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    # Fixation drops when sustain begins.
    _fix(trial, 0, cue_off)

    for i in range(B):
        trial.x[cue_on:cue_off, i, IN_REAL_A] = freqs[i] / f_max
        for t in range(cue_off, tdim):
            trial.y[t, i, OUT_REAL_A] = np.sin(2 * np.pi * freqs[i] * (t - cue_off))

    trial.add_cost_mask(response_on=cue_off)
    trial.epochs = {
        'fix1'   : (0,       cue_on),
        'cue'    : (cue_on,  cue_off),
        'sustain': (cue_off, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T17: SequenceRecall
# ---------------------------------------------------------------------------

def sequencerecall(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T17: SequenceRecall.

    K angles shown sequentially on channel A. After a delay, reproduce
    them in the same order.

    Inputs:  fixation, angles on channel A
    Outputs: fixation, angles reproduced sequentially
    """
    dt = config['dt']
    B  = config['batch_size']
    K  = rng.choice([3, 4])

    angles    = rng.uniform(0, 2 * np.pi, (B, K))
    fix_dur   = _ri(rng, 30, 80)
    item_dur  = _ri(rng, 40, 60)
    delay_dur = _ri(rng, 50, 150)
    resp_dur  = _ri(rng, 40, 60)   # per item

    enc_end  = fix_dur + K * item_dur
    resp_on  = enc_end + delay_dur
    tdim     = resp_on + K * resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)

    for k in range(K):
        t0 = fix_dur + k * item_dur
        t1 = t0 + item_dur
        for i in range(B):
            trial.x[t0:t1, i, IN_SIN_A] = np.sin(angles[i, k])
            trial.x[t0:t1, i, IN_COS_A] = np.cos(angles[i, k])

    for k in range(K):
        t0 = resp_on + k * resp_dur
        t1 = t0 + resp_dur
        for i in range(B):
            trial.y[t0:t1, i, OUT_SIN] = np.sin(angles[i, k])
            trial.y[t0:t1, i, OUT_COS] = np.cos(angles[i, k])

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,       fix_dur),
        'items' : (fix_dur, enc_end),
        'delay1': (enc_end, resp_on),
        'resp'  : (resp_on, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T18: Toggle
# ---------------------------------------------------------------------------

def toggle(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T18: Toggle (bistable flip-flop).

    Pulses on IN_REAL_A flip output between -1 and +1.
    Continuous output throughout — no separate response epoch.

    Inputs:  fixation during fix, pulses on IN_REAL_A
    Outputs: fixation during fix, toggle state on OUT_REAL_A
    """
    dt    = config['dt']
    B     = config['batch_size']

    fix_dur    = _ri(rng, 30, 80)
    stream_dur = _ri(rng, 200, 400)
    pulse_rate = _ru(rng, 0.01, 0.04)
    tdim       = fix_dur + stream_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, fix_dur)

    pulse_w = max(1, int(4 / dt))
    for i in range(B):
        state = rng.choice([-1.0, 1.0])
        trial.y[fix_dur:, i, OUT_REAL_A] = state
        t = fix_dur
        while t < tdim - pulse_w:
            if rng.rand() < pulse_rate:
                p_end = min(t + pulse_w, tdim)
                trial.x[t:p_end, i, IN_REAL_A] = 1.0
                state = -state
                trial.y[p_end:, i, OUT_REAL_A] = state
                t += pulse_w + max(1, _ri(rng, 20, 70))
            else:
                t += 1

    trial.add_cost_mask(response_on=fix_dur)
    trial.epochs = {
        'fix1'  : (0,       fix_dur),
        'stream': (fix_dur, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T19: ConditionalToggle
# ---------------------------------------------------------------------------

def conditionaltoggle(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T19: ConditionalToggle.

    Two independent pulse streams (IN_REAL_A, IN_REAL_B) toggle two
    independent outputs (OUT_REAL_A, OUT_REAL_B).

    Inputs:  fixation, pulses on IN_REAL_A and IN_REAL_B
    Outputs: fixation, state A on OUT_REAL_A, state B on OUT_REAL_B
    """
    dt    = config['dt']
    B     = config['batch_size']

    fix_dur    = _ri(rng, 30, 80)
    stream_dur = _ri(rng, 200, 400)
    rate_a     = _ru(rng, 0.01, 0.03)
    rate_b     = _ru(rng, 0.01, 0.03)
    tdim       = fix_dur + stream_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, fix_dur)

    pulse_w = max(1, int(4 / dt))
    for i in range(B):
        s_a = rng.choice([-1.0, 1.0])
        s_b = rng.choice([-1.0, 1.0])
        trial.y[fix_dur:, i, OUT_REAL_A] = s_a
        trial.y[fix_dur:, i, OUT_REAL_B] = s_b

        for ch, rate, out_ch in [
            (IN_REAL_A, rate_a, OUT_REAL_A),
            (IN_REAL_B, rate_b, OUT_REAL_B),
        ]:
            state = s_a if ch == IN_REAL_A else s_b
            t = fix_dur
            while t < tdim - pulse_w:
                if rng.rand() < rate:
                    p_end = min(t + pulse_w, tdim)
                    trial.x[t:p_end, i, ch] = 1.0
                    state = -state
                    trial.y[p_end:, i, out_ch] = state
                    t += pulse_w + max(1, _ri(rng, 20, 70))
                else:
                    t += 1

    trial.add_cost_mask(response_on=fix_dur)
    trial.epochs = {
        'fix1'  : (0,       fix_dur),
        'stream': (fix_dur, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T20: CueResponseAssoc
# ---------------------------------------------------------------------------

def cueresponseassoc(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T20: CueResponseAssoc (supervised proxy).

    Cue identity shown as a discrete angle on channel A.
    Fixed mapping: cue -> cue + pi/2.
    Full across-lifetime meta-learning version deferred to Phase 5.

    Inputs:  fixation, cue angle on channel A
    Outputs: fixation, associated response angle
    """
    dt     = config['dt']
    B      = config['batch_size']
    n_cues = 8

    cue_locs_all      = np.linspace(0, 2 * np.pi, n_cues, endpoint=False)
    response_locs_all = (cue_locs_all + np.pi / 2) % (2 * np.pi)

    cue_ids       = rng.choice(n_cues, size=B)
    cue_locs      = cue_locs_all[cue_ids]
    response_locs = response_locs_all[cue_ids]

    fix_dur  = _ri(rng, 30, 80)
    cue_dur  = _ri(rng, 30, 80)
    delay_dur= _ri(rng, 20, 80)
    resp_dur = _ri(rng, 30, 100)

    cue_on  = fix_dur
    cue_off = fix_dur + cue_dur
    resp_on = cue_off + delay_dur
    tdim    = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)
    _angle_in(trial, cue_locs, 'A', cue_on, cue_off)
    _angle_out(trial, response_locs, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,       cue_on),
        'cue1'  : (cue_on,  cue_off),
        'delay1': (cue_off, resp_on),
        'go1'   : (resp_on, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T21: PairedAssociation
# ---------------------------------------------------------------------------

def pairedassociation(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T21: PairedAssociation.

    Within one trial: PAIR_A (cue A on ch A + angle A on ch B),
    PAIR_B (cue B + angle B), PROBE (one cue alone), RESP (paired angle).
    Tests within-trial one-shot associative binding.

    Inputs:  fixation, cue angle on ch A, paired angle on ch B
    Outputs: fixation, recalled angle
    """
    dt     = config['dt']
    B      = config['batch_size']
    n_cues = 8

    cue_locs_all = np.linspace(0, 2 * np.pi, n_cues, endpoint=False)
    cue_ids_a    = rng.choice(n_cues, size=B)
    cue_ids_b    = rng.choice(n_cues, size=B)
    same         = cue_ids_b == cue_ids_a
    cue_ids_b[same] = (cue_ids_b[same] + 1) % n_cues

    cue_locs_a   = cue_locs_all[cue_ids_a]
    cue_locs_b   = cue_locs_all[cue_ids_b]
    angle_a      = rng.uniform(0, 2 * np.pi, B)
    angle_b      = rng.uniform(0, 2 * np.pi, B)
    probe_is_a   = rng.choice([0, 1], size=B).astype(bool)
    probe_locs   = np.where(probe_is_a, cue_locs_a, cue_locs_b)
    resp_locs    = np.where(probe_is_a, angle_a, angle_b)

    fix_dur    = _ri(rng, 30, 70)
    pair_dur   = _ri(rng, 40, 80)
    delay1_dur = _ri(rng, 30, 80)
    delay2_dur = _ri(rng, 30, 80)
    probe_dur  = _ri(rng, 30, 60)
    resp_dur   = _ri(rng, 30, 100)

    pair_a_on  = fix_dur
    pair_a_off = pair_a_on  + pair_dur
    pair_b_on  = pair_a_off + delay1_dur
    pair_b_off = pair_b_on  + pair_dur
    probe_on   = pair_b_off + delay2_dur
    probe_off  = probe_on   + probe_dur
    resp_on    = probe_off
    tdim       = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)

    _angle_in(trial, cue_locs_a, 'A', pair_a_on, pair_a_off)
    _angle_in(trial, angle_a,    'B', pair_a_on, pair_a_off)
    _angle_in(trial, cue_locs_b, 'A', pair_b_on, pair_b_off)
    _angle_in(trial, angle_b,    'B', pair_b_on, pair_b_off)
    _angle_in(trial, probe_locs, 'A', probe_on,  probe_off)
    _angle_out(trial, resp_locs, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,          pair_a_on),
        'pair_a': (pair_a_on,  pair_a_off),
        'delay1': (pair_a_off, pair_b_on),
        'pair_b': (pair_b_on,  pair_b_off),
        'delay2': (pair_b_off, probe_on),
        'probe' : (probe_on,   probe_off),
        'go1'   : (resp_on,    None),
    }
    return trial


# ---------------------------------------------------------------------------
# T22: ReversalLearning
# ---------------------------------------------------------------------------

def reversallearning(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T22: ReversalLearning.

    Like CueResponseAssoc but mapping reverses. Reversal signaled by
    IN_REAL_B = 1 during cue period (supervised proxy).

    Inputs:  fixation, cue angle on ch A, reversal flag on IN_REAL_B
    Outputs: fixation, response angle
    """
    dt     = config['dt']
    B      = config['batch_size']
    n_cues = 4

    cue_locs_all       = np.linspace(0, 2 * np.pi, n_cues, endpoint=False)
    response_locs_pre  = (cue_locs_all + np.pi / 2) % (2 * np.pi)
    response_locs_post = (response_locs_pre + np.pi) % (2 * np.pi)

    cue_ids       = rng.choice(n_cues, size=B)
    post_reversal = rng.choice([0, 1], size=B).astype(bool)
    cue_locs      = cue_locs_all[cue_ids]
    response_locs = np.where(
        post_reversal,
        response_locs_post[cue_ids],
        response_locs_pre[cue_ids])

    fix_dur   = _ri(rng, 30, 80)
    cue_dur   = _ri(rng, 30, 80)
    delay_dur = _ri(rng, 20, 80)
    resp_dur  = _ri(rng, 30, 100)

    cue_on  = fix_dur
    cue_off = fix_dur + cue_dur
    resp_on = cue_off + delay_dur
    tdim    = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)
    _angle_in(trial, cue_locs, 'A', cue_on, cue_off)

    for i in range(B):
        if post_reversal[i]:
            trial.x[cue_on:cue_off, i, IN_REAL_B] = 1.0

    _angle_out(trial, response_locs, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,       cue_on),
        'cue1'  : (cue_on,  cue_off),
        'delay1': (cue_off, resp_on),
        'go1'   : (resp_on, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T23: OnlineLinearReg
# ---------------------------------------------------------------------------

def onlinelinearreg(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T23: OnlineLinearReg.

    K examples of (x, ax+b) shown sequentially. Query x_q; predict y_q.
    Fresh function parameters per trial.

    Inputs:  fixation, x on IN_REAL_A, y on IN_REAL_B
    Outputs: fixation, predicted y on OUT_REAL_A
    """
    dt = config['dt']
    B  = config['batch_size']
    K  = _ri(rng, 5, 7)

    a        = rng.uniform(-1.0, 1.0, B)
    b        = rng.uniform(-0.5, 0.5, B)
    fix_dur  = _ri(rng, 30, 60)
    ex_dur   = _ri(rng, 30, 45)
    gap_dur  = max(1, int(8 / 1))
    probe_dur= _ri(rng, 35, 55)
    resp_dur = _ri(rng, 30, 70)

    examples_dur = K * (ex_dur + gap_dur)
    probe_on     = fix_dur + examples_dur
    probe_off    = probe_on + probe_dur
    resp_on      = probe_off
    tdim         = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)

    x_vals = rng.uniform(-0.8, 0.8, (B, K))
    for k in range(K):
        t0 = fix_dur + k * (ex_dur + gap_dur)
        t1 = t0 + ex_dur
        for i in range(B):
            y_val = np.clip(a[i] * x_vals[i, k] + b[i], -1.0, 1.0)
            trial.x[t0:t1, i, IN_REAL_A] = x_vals[i, k]
            trial.x[t0:t1, i, IN_REAL_B] = y_val

    x_q = rng.uniform(-0.8, 0.8, B)
    y_q = np.clip(a * x_q + b, -1.0, 1.0)
    for i in range(B):
        trial.x[probe_on:probe_off, i, IN_REAL_A] = x_q[i]
        trial.y[resp_on:, i, OUT_REAL_A] = y_q[i]

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'    : (0,        fix_dur),
        'examples': (fix_dur,  probe_on),
        'probe'   : (probe_on, probe_off),
        'go1'     : (resp_on,  None),
    }
    return trial


# ---------------------------------------------------------------------------
# T24: OnlineNonlinearReg
# ---------------------------------------------------------------------------

def onlinenonlinearreg(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T24: OnlineNonlinearReg.

    Like T23 but y = sin(omega * x + phi).

    Inputs:  fixation, x on IN_REAL_A, y on IN_REAL_B
    Outputs: fixation, predicted y on OUT_REAL_A
    """
    dt = config['dt']
    B  = config['batch_size']
    K  = _ri(rng, 7, 10)

    omega    = rng.uniform(0.8, 2.5, B)
    phi      = rng.uniform(0, 2 * np.pi, B)
    fix_dur  = _ri(rng, 30, 60)
    ex_dur   = _ri(rng, 25, 35)
    gap_dur  = max(1, int(6 / 1))
    probe_dur= _ri(rng, 35, 55)
    resp_dur = _ri(rng, 30, 70)

    examples_dur = K * (ex_dur + gap_dur)
    probe_on     = fix_dur + examples_dur
    probe_off    = probe_on + probe_dur
    resp_on      = probe_off
    tdim         = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)

    x_vals = rng.uniform(-1.0, 1.0, (B, K))
    for k in range(K):
        t0 = fix_dur + k * (ex_dur + gap_dur)
        t1 = t0 + ex_dur
        for i in range(B):
            y_val = np.sin(omega[i] * x_vals[i, k] + phi[i])
            trial.x[t0:t1, i, IN_REAL_A] = x_vals[i, k]
            trial.x[t0:t1, i, IN_REAL_B] = y_val

    x_q = rng.uniform(-1.0, 1.0, B)
    y_q = np.sin(omega * x_q + phi)
    for i in range(B):
        trial.x[probe_on:probe_off, i, IN_REAL_A] = x_q[i]
        trial.y[resp_on:, i, OUT_REAL_A] = y_q[i]

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'    : (0,        fix_dur),
        'examples': (fix_dur,  probe_on),
        'probe'   : (probe_on, probe_off),
        'go1'     : (resp_on,  None),
    }
    return trial


# ---------------------------------------------------------------------------
# T25: FewShotClassif
# ---------------------------------------------------------------------------

def fewshotclassif(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T25: FewShotClassif.

    K classes x M examples (feature vector on cue channels + label on
    IN_REAL_A). Query feature; output class label on OUT_REAL_A.

    Inputs:  fixation, features on cue channels 7-10, label on IN_REAL_A
    Outputs: fixation, class label on OUT_REAL_A
    """
    dt = config['dt']
    B  = config['batch_size']
    K  = 3
    M  = 3

    fix_dur   = _ri(rng, 30, 60)
    ex_dur    = _ri(rng, 30, 40)
    gap_dur   = max(1, int(6 / 1))
    delay_dur = _ri(rng, 40, 70)
    query_dur = _ri(rng, 40, 55)
    resp_dur  = _ri(rng, 30, 70)

    support_dur = K * M * (ex_dur + gap_dur)
    query_on    = fix_dur + support_dur + delay_dur
    query_off   = query_on + query_dur
    resp_on     = query_off
    tdim        = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)

    for i in range(B):
        prototypes = rng.uniform(-1, 1, (K, 4))
        support    = []
        for k in range(K):
            for m in range(M):
                feat = np.clip(prototypes[k] + rng.uniform(-0.15, 0.15, 4), -1, 1)
                support.append((feat, k))
        rng.shuffle(support)

        for idx, (feat, cls) in enumerate(support):
            t0 = fix_dur + idx * (ex_dur + gap_dur)
            t1 = t0 + ex_dur
            trial.x[t0:t1, i, IN_CUE_0:IN_CUE_3 + 1] = feat
            trial.x[t0:t1, i, IN_REAL_A] = (cls + 1) / K

        query_cls  = rng.randint(0, K)
        query_feat = np.clip(prototypes[query_cls] + rng.uniform(-0.15, 0.15, 4), -1, 1)
        trial.x[query_on:query_off, i, IN_CUE_0:IN_CUE_3 + 1] = query_feat
        trial.y[resp_on:, i, OUT_REAL_A] = (query_cls + 1) / K

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'   : (0,        fix_dur),
        'support': (fix_dur,  fix_dur + support_dur),
        'delay1' : (fix_dur + support_dur, query_on),
        'query'  : (query_on, query_off),
        'go1'    : (resp_on,  None),
    }
    return trial


# ---------------------------------------------------------------------------
# T26: MemoryDM
# ---------------------------------------------------------------------------

def memorydm(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T26: MemoryDM (compositional: memory + decision making).

    Store angle theta. Then observe evidence A_A and A_B.
    Respond at theta if A_A > A_B, else theta + pi.

    Inputs:  fixation, theta on ch A, evidence on IN_REAL_A/B
    Outputs: fixation, response angle
    """
    dt = config['dt']
    B  = config['batch_size']

    theta    = rng.uniform(0, 2 * np.pi, B)
    A_A      = rng.uniform(0.3, 1.0, B)
    A_B      = rng.uniform(0.3, 1.0, B)

    fix_dur   = _ri(rng, 30, 80)
    stim_dur  = _ri(rng, 30, 100)
    delay_dur = _ri(rng, 80, 200)
    ev_dur    = _ri(rng, 80, 200)
    resp_dur  = _ri(rng, 30, 100)

    stim_on  = fix_dur
    stim_off = stim_on + stim_dur
    ev_on    = stim_off + delay_dur
    ev_off   = ev_on + ev_dur
    resp_on  = ev_off
    tdim     = resp_on + resp_dur

    response_locs = np.where(A_A > A_B, theta, (theta + np.pi) % (2 * np.pi))

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)
    _angle_in(trial, theta, 'A', stim_on, stim_off)

    for i in range(B):
        trial.x[ev_on:ev_off, i, IN_REAL_A] = A_A[i]
        trial.x[ev_on:ev_off, i, IN_REAL_B] = A_B[i]

    _angle_out(trial, response_locs, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'    : (0,        stim_on),
        'stim1'   : (stim_on,  stim_off),
        'delay1'  : (stim_off, ev_on),
        'evidence': (ev_on,    ev_off),
        'go1'     : (resp_on,  None),
    }
    return trial


# ---------------------------------------------------------------------------
# T27: CountAndRecall
# ---------------------------------------------------------------------------

def countandrecall(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T27: CountAndRecall (compositional: counting + binding).

    N pulses on IN_REAL_A, each paired with an angle on ch A.
    Probe on IN_REAL_B indicates which pulse to recall.
    Output: angle of that pulse.

    Inputs:  fixation, pulses on IN_REAL_A, angles on ch A, probe on IN_REAL_B
    Outputs: fixation, recalled angle
    """
    dt    = config['dt']
    B     = config['batch_size']
    N_max = 4

    N_vals     = rng.randint(2, N_max + 1, size=B)
    fix_dur    = _ri(rng, 30, 80)
    stream_dur = _ri(rng, 200, 300)
    probe_dur  = _ri(rng, 40, 60)
    resp_dur   = _ri(rng, 30, 100)

    stream_on  = fix_dur
    stream_off = stream_on + stream_dur
    probe_on   = stream_off
    probe_off  = probe_on + probe_dur
    resp_on    = probe_off
    tdim       = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)

    pulse_w   = max(1, int(5 / 1))
    angle_dur = max(1, int(15 / 1))

    for i in range(B):
        N      = N_vals[i]
        angles = rng.uniform(0, 2 * np.pi, N)
        recall = rng.randint(0, N)
        spacing  = stream_dur // (N + 1)
        positions = [stream_on + spacing * (j + 1) for j in range(N)]
        positions = [min(p, stream_off - angle_dur - 5) for p in positions]

        for j, p in enumerate(positions):
            trial.x[p:min(p + pulse_w, stream_off), i, IN_REAL_A] = 1.0
            trial.x[p:min(p + angle_dur, stream_off), i, IN_SIN_A] = np.sin(angles[j])
            trial.x[p:min(p + angle_dur, stream_off), i, IN_COS_A] = np.cos(angles[j])

        trial.x[probe_on:probe_off, i, IN_REAL_B] = (recall + 1) / N
        trial.y[resp_on:, i, OUT_SIN] = np.sin(angles[recall])
        trial.y[resp_on:, i, OUT_COS] = np.cos(angles[recall])

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,          stream_on),
        'stream': (stream_on,  stream_off),
        'probe' : (probe_on,   probe_off),
        'go1'   : (resp_on,    None),
    }
    return trial


# ---------------------------------------------------------------------------
# T28: ConditionalRhythm
# ---------------------------------------------------------------------------

def conditionalrhythm(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T28: ConditionalRhythm (compositional: limit cycle + context gating).

    Frequency cue f1 and f2 given at start on IN_REAL_A/B.
    Network generates sin at f1, then switch pulse on IN_REAL_B triggers
    switch to f2.

    Inputs:  fixation, f1 on IN_REAL_A, f2 on IN_REAL_B during cue,
             switch pulse on IN_REAL_B during phase 1
    Outputs: fixation during cue, sinusoid on OUT_REAL_A
    """
    dt    = config['dt']
    B     = config['batch_size']
    f_min = 0.02
    f_max = 0.10

    f1          = rng.uniform(f_min, f_max / 2, B)
    f2          = rng.uniform(f_max / 2, f_max, B)
    fix_dur     = _ri(rng, 30, 60)
    cue_dur     = _ri(rng, 25, 35)
    phase1_dur  = _ri(rng, 180, 260)
    switch_dur  = max(1, int(5 / 1))
    phase2_dur  = _ri(rng, 180, 260)

    cue_on     = fix_dur
    cue_off    = cue_on + cue_dur
    phase1_on  = cue_off
    phase1_off = phase1_on + phase1_dur
    switch_on  = phase1_off
    switch_off = switch_on + switch_dur
    phase2_on  = switch_off
    phase2_off = phase2_on + phase2_dur
    tdim       = phase2_off

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, cue_off)

    for i in range(B):
        trial.x[cue_on:cue_off, i, IN_REAL_A] = f1[i] / f_max
        trial.x[cue_on:cue_off, i, IN_REAL_B] = f2[i] / f_max
        trial.x[switch_on:switch_off, i, IN_REAL_B] = 1.0
        for t in range(phase1_on, phase1_off):
            trial.y[t, i, OUT_REAL_A] = np.sin(2 * np.pi * f1[i] * (t - phase1_on))
        for t in range(phase2_on, phase2_off):
            trial.y[t, i, OUT_REAL_A] = np.sin(2 * np.pi * f2[i] * (t - phase2_on))

    trial.add_cost_mask(response_on=phase1_on)
    trial.epochs = {
        'fix1'   : (0,          cue_on),
        'cue_1'  : (cue_on,     cue_off),
        'phase_1': (phase1_on,  phase1_off),
        'switch' : (switch_on,  switch_off),
        'phase_2': (phase2_on,  phase2_off),
    }
    return trial


# ---------------------------------------------------------------------------
# T29: DelayedAssociation
# ---------------------------------------------------------------------------

def delayedassociation(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T29: DelayedAssociation (compositional: binding + interference resistance).

    See (cue angle on ch A, paired angle on ch B). Then distractors on ch A.
    Probe: cue shown alone. Recall paired angle.

    Inputs:  fixation, cue on ch A, paired angle on ch B,
             distractor angles on ch A during distract period
    Outputs: fixation, recalled paired angle
    """
    dt     = config['dt']
    B      = config['batch_size']
    n_cues = 8

    cue_locs_all  = np.linspace(0, 2 * np.pi, n_cues, endpoint=False)
    cue_ids       = rng.choice(n_cues, size=B)
    cue_locs      = cue_locs_all[cue_ids]
    paired_angles = rng.uniform(0, 2 * np.pi, B)

    fix_dur      = _ri(rng, 30, 60)
    pair_dur     = _ri(rng, 40, 80)
    distract_dur = _ri(rng, 250, 400)
    probe_dur    = _ri(rng, 40, 60)
    resp_dur     = _ri(rng, 30, 100)
    n_distractors= _ri(rng, 4, 9)

    pair_on      = fix_dur
    pair_off     = pair_on + pair_dur
    distract_on  = pair_off
    distract_off = distract_on + distract_dur
    probe_on     = distract_off
    probe_off    = probe_on + probe_dur
    resp_on      = probe_off
    tdim         = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)
    _angle_in(trial, cue_locs,      'A', pair_on, pair_off)
    _angle_in(trial, paired_angles, 'B', pair_on, pair_off)

    # Distractor angles on ch A at irregular times.
    spacing = distract_dur // (n_distractors + 1)
    for j in range(n_distractors):
        d_on  = distract_on + j * spacing + _ri(rng, 0, spacing // 3)
        d_dur = _ri(rng, 20, 40)
        d_off = min(d_on + d_dur, distract_off - 5)
        if d_off > d_on:
            d_locs = rng.uniform(0, 2 * np.pi, B)
            _angle_in(trial, d_locs, 'A', d_on, d_off)

    _angle_in(trial,  cue_locs,      'A', probe_on, probe_off)
    _angle_out(trial, paired_angles,       resp_on,  tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'    : (0,           pair_on),
        'pair'    : (pair_on,     pair_off),
        'distract': (distract_on, distract_off),
        'probe'   : (probe_on,    probe_off),
        'go1'     : (resp_on,     None),
    }
    return trial


# ---------------------------------------------------------------------------
# T30: SequentialDecision
# ---------------------------------------------------------------------------

def sequentialdecision(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T30: SequentialDecision (compositional: evidence integration x3).

    Three sequential decision blocks. Each block: two stimuli on ch A and B;
    stronger wins. Final response is circular mean of three winners.

    Inputs:  fixation, stimuli on ch A and B across three blocks
    Outputs: fixation, composite response angle
    """
    dt       = config['dt']
    B        = config['batch_size']
    n_blocks = 3

    fix_dur  = _ri(rng, 30, 80)
    block_dur= _ri(rng, 90, 130)
    resp_dur = _ri(rng, 30, 100)

    resp_on  = fix_dur + n_blocks * block_dur
    tdim     = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])
    _fix(trial, 0, resp_on)

    sin_sum = np.zeros(B)
    cos_sum = np.zeros(B)

    for blk in range(n_blocks):
        blk_on  = fix_dur + blk * block_dur
        blk_off = blk_on + block_dur

        theta_A = rng.uniform(0, 2 * np.pi, B)
        theta_B = rng.uniform(0, 2 * np.pi, B)
        amp_A   = rng.uniform(0.3, 1.0, B)
        amp_B   = rng.uniform(0.3, 1.0, B)

        for i in range(B):
            trial.x[blk_on:blk_off, i, IN_SIN_A] = amp_A[i] * np.sin(theta_A[i])
            trial.x[blk_on:blk_off, i, IN_COS_A] = amp_A[i] * np.cos(theta_A[i])
            trial.x[blk_on:blk_off, i, IN_SIN_B] = amp_B[i] * np.sin(theta_B[i])
            trial.x[blk_on:blk_off, i, IN_COS_B] = amp_B[i] * np.cos(theta_B[i])

        winner  = np.where(amp_A > amp_B, theta_A, theta_B)
        sin_sum += np.sin(winner)
        cos_sum += np.cos(winner)

    composite = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)
    _angle_out(trial, composite, resp_on, tdim)

    ep = {'fix1': (0, fix_dur)}
    for blk in range(n_blocks):
        ep[f'block_{blk+1}'] = (fix_dur + blk * block_dur,
                                 fix_dur + (blk + 1) * block_dur)
    ep['go1'] = (resp_on, None)
    trial.epochs = ep

    trial.add_cost_mask(response_on=resp_on)
    return trial


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

NEW_TASKS = {
    'multiitemrecall'    : multiitemrecall,    # T12
    'pulsecounting'      : pulsecounting,      # T13
    'intervalreproduction': intervalreproduction, # T14
    'pulserateestimation': pulserateestimation, # T15
    'rhythmgeneration'   : rhythmgeneration,   # T16
    'sequencerecall'     : sequencerecall,     # T17
    'toggle'             : toggle,             # T18
    'conditionaltoggle'  : conditionaltoggle,  # T19
    'cueresponseassoc'   : cueresponseassoc,   # T20
    'pairedassociation'  : pairedassociation,  # T21
    'reversallearning'   : reversallearning,   # T22
    'onlinelinearreg'    : onlinelinearreg,    # T23
    'onlinenonlinearreg' : onlinenonlinearreg, # T24
    'fewshotclassif'     : fewshotclassif,     # T25
    'memorydm'           : memorydm,           # T26
    'countandrecall'     : countandrecall,     # T27
    'conditionalrhythm'  : conditionalrhythm,  # T28
    'delayedassociation' : delayedassociation, # T29
    'sequentialdecision' : sequentialdecision, # T30
}

NEW_TASK_NAMES = {
    'multiitemrecall'    : 'Multi-Item Recall',
    'pulsecounting'      : 'Pulse Count',
    'intervalreproduction': 'Interval Repro',
    'pulserateestimation': 'Rate Estim',
    'rhythmgeneration'   : 'Rhythm Gen',
    'sequencerecall'     : 'Seq Recall',
    'toggle'             : 'Toggle',
    'conditionaltoggle'  : 'Cond Toggle',
    'cueresponseassoc'   : 'Cue Assoc',
    'pairedassociation'  : 'Paired Assoc',
    'reversallearning'   : 'Reversal Learn',
    'onlinelinearreg'    : 'Online Lin Reg',
    'onlinenonlinearreg' : 'Online Nonlin Reg',
    'fewshotclassif'     : 'Few-Shot Classif',
    'memorydm'           : 'Memory DM',
    'countandrecall'     : 'Count & Recall',
    'conditionalrhythm'  : 'Cond Rhythm',
    'delayedassociation' : 'Delayed Assoc',
    'sequentialdecision' : 'Sequential Dec',
}