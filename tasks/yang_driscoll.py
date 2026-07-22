"""
Input channels used:
    IN_FIX    (0)  - fixation signal
    IN_SIN_A  (1)  - sin of angle A
    IN_COS_A  (2)  - cos of angle A
    IN_SIN_B  (3)  - sin of angle B
    IN_COS_B  (4)  - cos of angle B
    IN_REAL_A (5)  - scalar input A (amplitude, evidence)
    IN_REAL_B (6)  - scalar input B

Output channels used:
    OUT_FIX   (0)  - fixation hold signal
    OUT_SIN   (1)  - sin of response angle
    OUT_COS   (2)  - cos of response angle
"""

from __future__ import annotations

import numpy as np
from tasks.base import (
    Trial,
    IN_FIX, IN_SIN_A, IN_COS_A, IN_SIN_B, IN_COS_B,
    IN_REAL_A, IN_REAL_B,
    OUT_FIX, OUT_SIN, OUT_COS,
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


def _set_angle_in(trial, angles, channel, t0, t1):
    """Write sin/cos of angles into input channels over [t0, t1)."""
    if channel == 'A':
        trial.x[t0:t1, :, IN_SIN_A] = np.sin(angles)
        trial.x[t0:t1, :, IN_COS_A] = np.cos(angles)
    else:
        trial.x[t0:t1, :, IN_SIN_B] = np.sin(angles)
        trial.x[t0:t1, :, IN_COS_B] = np.cos(angles)


def _set_angle_out(trial, angles, t0, t1):
    """Write sin/cos of angles into output channels over [t0, t1)."""
    trial.y[t0:t1, :, OUT_SIN] = np.sin(angles)
    trial.y[t0:t1, :, OUT_COS] = np.cos(angles)


def _set_fixation(trial, t0, t1):
    """Set fixation input and output to 1 over [t0, t1)."""
    trial.x[t0:t1, :, IN_FIX]  = 1.0
    trial.y[t0:t1, :, OUT_FIX] = 1.0


# ---------------------------------------------------------------------------
# T01: DelayPro
# ---------------------------------------------------------------------------

def delaypro(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T01: DelayPro.

    See an angle on channel A. Stimulus stays visible throughout.
    At go cue (fixation drops), respond at that same angle.
    No memory required — stimulus is always present.

    Epochs: FIX | STIM | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    angles   = rng.uniform(0, 2 * np.pi, B)
    fix_dur  = _ri(rng, 30, 80)
    stim_dur = _ri(rng, 30, 150)
    resp_dur = _ri(rng, 30, 100)
    tdim     = fix_dur + stim_dur + resp_dur

    stim_on  = fix_dur
    stim_off = fix_dur + stim_dur   # stimulus stays on through resp too
    resp_on  = fix_dur + stim_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)
    # Stimulus visible from stim_on through end of trial.
    _set_angle_in(trial, angles, 'A', stim_on, tdim)
    # Response: same angle.
    _set_angle_out(trial, angles, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1' : (0,        stim_on),
        'stim1': (stim_on,  resp_on),
        'go1'  : (resp_on,  None),
    }
    return trial


# ---------------------------------------------------------------------------
# T02: DelayAnti
# ---------------------------------------------------------------------------

def delayanti(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T02: DelayAnti.

    Same as DelayPro but respond at the opposite angle (theta + pi).
    Stimulus stays visible throughout.

    Epochs: FIX | STIM | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    angles   = rng.uniform(0, 2 * np.pi, B)
    fix_dur  = _ri(rng, 30, 80)
    stim_dur = _ri(rng, 30, 150)
    resp_dur = _ri(rng, 30, 100)

    stim_on = fix_dur
    resp_on = fix_dur + stim_dur
    tdim    = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)
    _set_angle_in(trial, angles, 'A', stim_on, tdim)
    _set_angle_out(trial, (angles + np.pi) % (2 * np.pi), resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1' : (0,       stim_on),
        'stim1': (stim_on, resp_on),
        'go1'  : (resp_on, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T03: MemoryPro
# ---------------------------------------------------------------------------

def memorypro(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T03: MemoryPro.

    See an angle briefly, then it disappears. Hold it across a delay.
    At go cue, respond at that angle.

    Epochs: FIX | STIM | DELAY | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    angles    = rng.uniform(0, 2 * np.pi, B)
    fix_dur   = _ri(rng, 30, 80)
    stim_dur  = _ri(rng, 30, 100)
    delay_dur = _ri(rng, 50, 300)
    resp_dur  = _ri(rng, 30, 100)

    stim_on  = fix_dur
    stim_off = fix_dur + stim_dur
    resp_on  = stim_off + delay_dur
    tdim     = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)
    # Stimulus shown briefly then removed.
    _set_angle_in(trial, angles, 'A', stim_on, stim_off)
    _set_angle_out(trial, angles, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,        stim_on),
        'stim1' : (stim_on,  stim_off),
        'delay1': (stim_off, resp_on),
        'go1'   : (resp_on,  None),
    }
    return trial


# ---------------------------------------------------------------------------
# T04: MemoryAnti
# ---------------------------------------------------------------------------

def memoryanti(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T04: MemoryAnti.

    Like MemoryPro but respond at the opposite angle (theta + pi).

    Epochs: FIX | STIM | DELAY | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    angles    = rng.uniform(0, 2 * np.pi, B)
    fix_dur   = _ri(rng, 30, 80)
    stim_dur  = _ri(rng, 30, 100)
    delay_dur = _ri(rng, 50, 300)
    resp_dur  = _ri(rng, 30, 100)

    stim_on  = fix_dur
    stim_off = fix_dur + stim_dur
    resp_on  = stim_off + delay_dur
    tdim     = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)
    _set_angle_in(trial, angles, 'A', stim_on, stim_off)
    _set_angle_out(trial, (angles + np.pi) % (2 * np.pi), resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,        stim_on),
        'stim1' : (stim_on,  stim_off),
        'delay1': (stim_off, resp_on),
        'go1'   : (resp_on,  None),
    }
    return trial


# ---------------------------------------------------------------------------
# T05: DM (Decision Making)
# ---------------------------------------------------------------------------

def dm(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T05: DM.

    Two noisy angle stimuli shown simultaneously on channels A and B,
    each with a scalar amplitude. Respond at the angle of the stronger
    (higher amplitude) stimulus.

    Amplitude is encoded as the magnitude of the sin/cos inputs:
        x[IN_SIN_A] = amp_A * sin(theta_A)
        x[IN_COS_A] = amp_A * cos(theta_A)

    Epochs: FIX | STIM | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    theta_A  = rng.uniform(0, 2 * np.pi, B)
    theta_B  = rng.uniform(0, 2 * np.pi, B)
    amp_A    = rng.uniform(0.3, 1.0, B)
    amp_B    = rng.uniform(0.3, 1.0, B)

    fix_dur  = _ri(rng, 30, 80)
    stim_dur = _ri(rng, 80, 200)
    resp_dur = _ri(rng, 30, 100)

    stim_on = fix_dur
    resp_on = fix_dur + stim_dur
    tdim    = resp_on + resp_dur

    # Winner: higher amplitude.
    winner = np.where(amp_A > amp_B, theta_A, theta_B)

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)

    # Amplitude-modulated sin/cos inputs.
    trial.x[stim_on:resp_on, :, IN_SIN_A] = amp_A * np.sin(theta_A)
    trial.x[stim_on:resp_on, :, IN_COS_A] = amp_A * np.cos(theta_A)
    trial.x[stim_on:resp_on, :, IN_SIN_B] = amp_B * np.sin(theta_B)
    trial.x[stim_on:resp_on, :, IN_COS_B] = amp_B * np.cos(theta_B)

    _set_angle_out(trial, winner, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1' : (0,       stim_on),
        'stim1': (stim_on, resp_on),
        'go1'  : (resp_on, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T06: DMAnti
# ---------------------------------------------------------------------------

def dmanti(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T06: DMAnti.

    Like DM but respond at the angle of the WEAKER stimulus.

    Epochs: FIX | STIM | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    theta_A  = rng.uniform(0, 2 * np.pi, B)
    theta_B  = rng.uniform(0, 2 * np.pi, B)
    amp_A    = rng.uniform(0.3, 1.0, B)
    amp_B    = rng.uniform(0.3, 1.0, B)

    fix_dur  = _ri(rng, 30, 80)
    stim_dur = _ri(rng, 80, 200)
    resp_dur = _ri(rng, 30, 100)

    stim_on = fix_dur
    resp_on = fix_dur + stim_dur
    tdim    = resp_on + resp_dur

    # Loser: lower amplitude.
    loser = np.where(amp_A < amp_B, theta_A, theta_B)

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)
    trial.x[stim_on:resp_on, :, IN_SIN_A] = amp_A * np.sin(theta_A)
    trial.x[stim_on:resp_on, :, IN_COS_A] = amp_A * np.cos(theta_A)
    trial.x[stim_on:resp_on, :, IN_SIN_B] = amp_B * np.sin(theta_B)
    trial.x[stim_on:resp_on, :, IN_COS_B] = amp_B * np.cos(theta_B)

    _set_angle_out(trial, loser, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1' : (0,       stim_on),
        'stim1': (stim_on, resp_on),
        'go1'  : (resp_on, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T07: ContextDM-A
# ---------------------------------------------------------------------------

def contextdm_a(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T07: ContextDM-A.

    Two stimuli on both channels A and B with independent amplitudes.
    Task context (encoded in task_vec, not in the inputs) says: attend
    to channel A only. Respond at the angle of channel A regardless of B.

    Channel B is a distractor.

    Epochs: FIX | STIM | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    theta_A   = rng.uniform(0, 2 * np.pi, B)
    theta_B   = rng.uniform(0, 2 * np.pi, B)
    amp_A     = rng.uniform(0.3, 1.0, B)
    amp_B     = rng.uniform(0.3, 1.0, B)

    fix_dur  = _ri(rng, 30, 80)
    stim_dur = _ri(rng, 80, 200)
    resp_dur = _ri(rng, 30, 100)

    stim_on = fix_dur
    resp_on = fix_dur + stim_dur
    tdim    = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)
    trial.x[stim_on:resp_on, :, IN_SIN_A] = amp_A * np.sin(theta_A)
    trial.x[stim_on:resp_on, :, IN_COS_A] = amp_A * np.cos(theta_A)
    trial.x[stim_on:resp_on, :, IN_SIN_B] = amp_B * np.sin(theta_B)
    trial.x[stim_on:resp_on, :, IN_COS_B] = amp_B * np.cos(theta_B)

    # Always respond at channel A angle.
    _set_angle_out(trial, theta_A, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1' : (0,       stim_on),
        'stim1': (stim_on, resp_on),
        'go1'  : (resp_on, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T08: ContextDM-B
# ---------------------------------------------------------------------------

def contextdm_b(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T08: ContextDM-B.

    Same inputs as ContextDM-A but attend to channel B.
    Channel A is the distractor.

    Epochs: FIX | STIM | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    theta_A   = rng.uniform(0, 2 * np.pi, B)
    theta_B   = rng.uniform(0, 2 * np.pi, B)
    amp_A     = rng.uniform(0.3, 1.0, B)
    amp_B     = rng.uniform(0.3, 1.0, B)

    fix_dur  = _ri(rng, 30, 80)
    stim_dur = _ri(rng, 80, 200)
    resp_dur = _ri(rng, 30, 100)

    stim_on = fix_dur
    resp_on = fix_dur + stim_dur
    tdim    = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)
    trial.x[stim_on:resp_on, :, IN_SIN_A] = amp_A * np.sin(theta_A)
    trial.x[stim_on:resp_on, :, IN_COS_A] = amp_A * np.cos(theta_A)
    trial.x[stim_on:resp_on, :, IN_SIN_B] = amp_B * np.sin(theta_B)
    trial.x[stim_on:resp_on, :, IN_COS_B] = amp_B * np.cos(theta_B)

    # Always respond at channel B angle.
    _set_angle_out(trial, theta_B, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1' : (0,       stim_on),
        'stim1': (stim_on, resp_on),
        'go1'  : (resp_on, None),
    }
    return trial


# ---------------------------------------------------------------------------
# T09: DelayMatchSample
# ---------------------------------------------------------------------------

def delaymatchsample(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T09: DelayMatchSample.

    See theta_1. After a delay, see theta_2.
    If they match (angular distance < threshold): respond at theta_1.
    If they don't match: respond at theta_1 + pi.

    Epochs: FIX | STIM_1 | DELAY | STIM_2 | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    match_threshold = np.pi / 6   # 30 degrees

    theta_1   = rng.uniform(0, 2 * np.pi, B)
    is_match  = rng.choice([0, 1], size=B).astype(bool)
    # Non-match: offset by pi +/- some jitter.
    offset    = np.where(
        is_match,
        rng.uniform(-0.2, 0.2, B),
        np.pi + rng.uniform(-0.5, 0.5, B))
    theta_2   = (theta_1 + offset) % (2 * np.pi)

    fix_dur   = _ri(rng, 30, 80)
    stim1_dur = _ri(rng, 30, 100)
    delay_dur = _ri(rng, 50, 200)
    stim2_dur = _ri(rng, 30, 100)
    resp_dur  = _ri(rng, 30, 100)

    stim1_on  = fix_dur
    stim1_off = stim1_on + stim1_dur
    stim2_on  = stim1_off + delay_dur
    stim2_off = stim2_on + stim2_dur
    resp_on   = stim2_off
    tdim      = resp_on + resp_dur

    # Response: theta_1 if match, theta_1+pi if non-match.
    response = np.where(is_match, theta_1, (theta_1 + np.pi) % (2 * np.pi))

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)
    _set_angle_in(trial, theta_1, 'A', stim1_on, stim1_off)
    _set_angle_in(trial, theta_2, 'A', stim2_on, stim2_off)
    _set_angle_out(trial, response, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,         stim1_on),
        'stim1' : (stim1_on,  stim1_off),
        'delay1': (stim1_off, stim2_on),
        'stim2' : (stim2_on,  stim2_off),
        'go1'   : (resp_on,   None),
    }
    return trial


# ---------------------------------------------------------------------------
# T10: DelayNonMatchSample
# ---------------------------------------------------------------------------

def delaynonmatchsample(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T10: DelayNonMatchSample.

    Inverted convention from T09:
    If they match: respond at theta_1 + pi.
    If they don't match: respond at theta_1.

    Epochs: FIX | STIM_1 | DELAY | STIM_2 | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    theta_1   = rng.uniform(0, 2 * np.pi, B)
    is_match  = rng.choice([0, 1], size=B).astype(bool)
    offset    = np.where(
        is_match,
        rng.uniform(-0.2, 0.2, B),
        np.pi + rng.uniform(-0.5, 0.5, B))
    theta_2   = (theta_1 + offset) % (2 * np.pi)

    fix_dur   = _ri(rng, 30, 80)
    stim1_dur = _ri(rng, 30, 100)
    delay_dur = _ri(rng, 50, 200)
    stim2_dur = _ri(rng, 30, 100)
    resp_dur  = _ri(rng, 30, 100)

    stim1_on  = fix_dur
    stim1_off = stim1_on + stim1_dur
    stim2_on  = stim1_off + delay_dur
    stim2_off = stim2_on + stim2_dur
    resp_on   = stim2_off
    tdim      = resp_on + resp_dur

    # Inverted: non-match -> respond at theta_1, match -> theta_1+pi.
    response = np.where(~is_match, theta_1, (theta_1 + np.pi) % (2 * np.pi))

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)
    _set_angle_in(trial, theta_1, 'A', stim1_on, stim1_off)
    _set_angle_in(trial, theta_2, 'A', stim2_on, stim2_off)
    _set_angle_out(trial, response, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'  : (0,         stim1_on),
        'stim1' : (stim1_on,  stim1_off),
        'delay1': (stim1_off, stim2_on),
        'stim2' : (stim2_on,  stim2_off),
        'go1'   : (resp_on,   None),
    }
    return trial


# ---------------------------------------------------------------------------
# T11: ExtendedMemory
# ---------------------------------------------------------------------------

def extendedmemory(config: dict, rng: np.random.RandomState) -> Trial:
    """
    T11: ExtendedMemory.

    Identical to MemoryPro (T03) but with a much longer delay period
    (500-1000 timesteps instead of 50-300). Tests how memory degrades
    over time and whether the network uses activity-based vs
    weight-based storage.

    Epochs: FIX | STIM | LONG_DELAY | RESP
    """
    dt = config['dt']
    B  = config['batch_size']

    angles    = rng.uniform(0, 2 * np.pi, B)
    fix_dur   = _ri(rng, 30, 80)
    stim_dur  = _ri(rng, 30, 100)
    delay_dur = _ri(rng, 500, 1000)   # long delay
    resp_dur  = _ri(rng, 30, 100)

    stim_on  = fix_dur
    stim_off = fix_dur + stim_dur
    resp_on  = stim_off + delay_dur
    tdim     = resp_on + resp_dur

    trial = Trial(tdim, B, dt=dt,
                  sigma_x=config['sigma_x'], alpha=config['alpha'])

    _set_fixation(trial, 0, resp_on)
    _set_angle_in(trial, angles, 'A', stim_on, stim_off)
    _set_angle_out(trial, angles, resp_on, tdim)

    trial.add_cost_mask(response_on=resp_on)
    trial.epochs = {
        'fix1'      : (0,        stim_on),
        'stim1'     : (stim_on,  stim_off),
        'long_delay': (stim_off, resp_on),
        'go1'       : (resp_on,  None),
    }
    return trial


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

YANG_DRISCOLL_TASKS = {
    'delaypro'           : delaypro,           # T01
    'delayanti'          : delayanti,           # T02
    'memorypro'          : memorypro,           # T03
    'memoryanti'         : memoryanti,          # T04
    'dm'                 : dm,                  # T05
    'dmanti'             : dmanti,              # T06
    'contextdm_a'        : contextdm_a,         # T07
    'contextdm_b'        : contextdm_b,         # T08
    'delaymatchsample'   : delaymatchsample,    # T09
    'delaynonmatchsample': delaynonmatchsample, # T10
    'extendedmemory'     : extendedmemory,      # T11
}

YANG_DRISCOLL_NAMES = {
    'delaypro'           : 'Delay Pro',
    'delayanti'          : 'Delay Anti',
    'memorypro'          : 'Memory Pro',
    'memoryanti'         : 'Memory Anti',
    'dm'                 : 'DM',
    'dmanti'             : 'DM Anti',
    'contextdm_a'        : 'Context DM-A',
    'contextdm_b'        : 'Context DM-B',
    'delaymatchsample'   : 'DMS',
    'delaynonmatchsample': 'DNMS',
    'extendedmemory'     : 'Ext Memory',
}
