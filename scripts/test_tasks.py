"""
scripts/test_tasks.py

Sanity check for the meta-plastic-rnn codebase.
Run this from the repo root before any training to confirm:
    1. All Yang/Driscoll tasks generate correctly shaped trials
    2. Input/output values are in expected ranges
    3. Cost mask is non-zero in the response epoch
    4. The LeakyRNN forward pass works end-to-end
    5. The GRU forward pass works end-to-end
    6. One training step runs without errors

Usage:
    cd ~/projects/meta-plastic-rnn
    python scripts/test_tasks.py

Expected output: all checks print PASS. Any FAIL means something
needs fixing before running a real training job.
"""

import sys
import traceback
import numpy as np
import torch
from tasks.new_tasks import NEW_TASKS, NEW_TASK_NAMES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    msg    = f'  [{status}] {name}'
    if detail:
        msg += f'  ({detail})'
    print(msg)
    return condition


def section(name):
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')


# ---------------------------------------------------------------------------
# Import everything
# ---------------------------------------------------------------------------

section('Imports')

try:
    from tasks.base import (
        Trial, TaskDataset, make_config, collate_trials,
        N_INPUT, N_OUTPUT,
        IN_FIX, IN_SIN_A, IN_COS_A, OUT_FIX, OUT_SIN, OUT_COS,
    )
    check('tasks.base', True)
except Exception as e:
    check('tasks.base', False, str(e))
    sys.exit(1)

try:
    from tasks.yang_driscoll import YANG_DRISCOLL_TASKS, YANG_DRISCOLL_NAMES
    check('tasks.yang_driscoll', True)
except Exception as e:
    check('tasks.yang_driscoll', False, str(e))
    sys.exit(1)

try:
    from network.rnn import LeakyRNN, GRUNet, build_network
    check('network.rnn', True)
except Exception as e:
    check('network.rnn', False, str(e))
    sys.exit(1)

try:
    from network.train import Trainer, masked_mse_loss, compute_performance
    check('network.train', True)
except Exception as e:
    check('network.train', False, str(e))
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

section('Config')

config = make_config(
    n_rnn       = 64,    # small for fast testing
    batch_size  = 8,
    sigma_x     = 0.1,
    sigma_rec   = 0.05,
)
rng = np.random.RandomState(42)

check('N_INPUT == 11',  N_INPUT  == 11, f'got {N_INPUT}')
check('N_OUTPUT == 5',  N_OUTPUT == 5,  f'got {N_OUTPUT}')
check('alpha == dt/tau',
      abs(config['alpha'] - config['dt'] / config['tau']) < 1e-6,
      f"alpha={config['alpha']:.4f}")
check('n_input in config',  config['n_input']  == N_INPUT)
check('n_output in config', config['n_output'] == N_OUTPUT)


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

section('Yang/Driscoll Task Generation')

all_passed = True
for task_name, task_fn in YANG_DRISCOLL_TASKS.items():
    try:
        trial = task_fn(config, rng)

        # Shape checks.
        ok_x = trial.x.shape == (trial.tdim, config['batch_size'], N_INPUT)
        ok_y = trial.y.shape == (trial.tdim, config['batch_size'], N_OUTPUT)
        ok_m = trial.c_mask.shape == (trial.tdim, config['batch_size'], N_OUTPUT)

        # Value range checks.
        ok_x_range = (trial.x >= -2.0).all() and (trial.x <= 2.0).all()
        ok_y_range = (trial.y >= -1.5).all() and (trial.y <= 1.5).all()

        # Fixation check: input fix should be 1 at start.
        ok_fix_in  = trial.x[0, 0, IN_FIX] == 1.0

        # Cost mask check: should be nonzero somewhere.
        ok_mask = trial.c_mask.sum() > 0

        # Epochs check: should be a non-empty dict.
        ok_epochs = isinstance(trial.epochs, dict) and len(trial.epochs) > 0

        # Noise check.
        trial.add_input_noise(rng)
        ok_noise = trial.x.shape == (trial.tdim, config['batch_size'], N_INPUT)

        # Tensor conversion.
        x, y, c = trial.to_tensors()
        ok_tensors = (
            isinstance(x, torch.Tensor) and
            x.shape == (trial.tdim, config['batch_size'], N_INPUT)
        )

        all_ok = all([ok_x, ok_y, ok_m, ok_x_range, ok_y_range,
                      ok_fix_in, ok_mask, ok_epochs, ok_noise, ok_tensors])

        detail = (f'T={trial.tdim} B={config["batch_size"]} '
                  f'epochs={list(trial.epochs.keys())}')
        passed = check(f'{task_name:<22}', all_ok, detail)

        if not all_ok:
            all_passed = False
            if not ok_x:      print(f'    x shape: {trial.x.shape}')
            if not ok_y:      print(f'    y shape: {trial.y.shape}')
            if not ok_x_range:print(f'    x range: [{trial.x.min():.2f}, {trial.x.max():.2f}]')
            if not ok_y_range:print(f'    y range: [{trial.y.min():.2f}, {trial.y.max():.2f}]')
            if not ok_fix_in: print(f'    fix_in at t=0: {trial.x[0, 0, IN_FIX]}')
            if not ok_mask:   print(f'    c_mask sum: {trial.c_mask.sum()}')

    except Exception as e:
        check(f'{task_name:<22}', False, str(e))
        traceback.print_exc()
        all_passed = False

print(f'\n  Yang/Driscoll tasks: {"all passed" if all_passed else "some failed"}')


# ---------------------------------------------------------------------------
# TaskDataset
# ---------------------------------------------------------------------------

section('TaskDataset')

try:
    dataset = TaskDataset(
        task_funcs        = YANG_DRISCOLL_TASKS,
        config            = config,
        batches_per_epoch = 10,
        seed              = 0,
    )
    check('TaskDataset created', True, f'{len(dataset)} batches per epoch')

    # Sample a few items.
    sample_ok = True
    for i in range(3):
        x, y, c_mask, task_name = dataset[i]
        if not (isinstance(x, torch.Tensor) and x.ndim == 3):
            sample_ok = False
    check('TaskDataset sampling', sample_ok,
          f'last sample: task={task_name} x={tuple(x.shape)}')

    # Curriculum subset.
    dataset.set_tasks(['delaypro', 'delayanti'])
    x, y, c_mask, task_name = dataset[0]
    check('set_tasks curriculum', task_name in ['delaypro', 'delayanti'],
          f'got task: {task_name}')
    dataset.set_tasks(list(YANG_DRISCOLL_TASKS.keys()))  # reset

except Exception as e:
    check('TaskDataset', False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Network forward pass
# ---------------------------------------------------------------------------

section('Network: LeakyRNN Forward Pass')

try:
    model = LeakyRNN(
        n_input     = N_INPUT,
        n_rnn       = config['n_rnn'],
        n_output    = N_OUTPUT,
        alpha       = config['alpha'],
        sigma_rec   = config['sigma_rec'],
        activation  = 'softplus',
        w_rec_init  = 'diag',
        w_rec_coeff = 1.0,
        l2_h        = 1e-6,
        l2_weight   = 1e-6,
    )
    check('LeakyRNN created', True,
          f'{sum(p.numel() for p in model.parameters()):,} params')

    # Generate a trial and run forward.
    trial    = YANG_DRISCOLL_TASKS['memorypro'](config, rng)
    x, y, c  = trial.to_tensors()

    model.eval()
    with torch.no_grad():
        result = model(x)

    check('forward pass runs',    True)
    check('output shape correct',
          result.output.shape == (trial.tdim, config['batch_size'], N_OUTPUT),
          f'{tuple(result.output.shape)}')
    check('hidden shape correct',
          result.hidden.shape == (trial.tdim, config['batch_size'], config['n_rnn']),
          f'{tuple(result.hidden.shape)}')
    check('loss_reg is scalar',   result.loss_reg.ndim == 0)
    check('output values finite', torch.isfinite(result.output).all().item())
    check('hidden values finite', torch.isfinite(result.hidden).all().item())

    # Training mode (adds recurrent noise).
    model.train()
    result_train = model(x)
    check('forward pass in train mode', torch.isfinite(result_train.output).all().item())

except Exception as e:
    check('LeakyRNN forward pass', False, str(e))
    traceback.print_exc()


section('Network: GRU Forward Pass')

try:
    gru = GRUNet(
        n_input    = N_INPUT,
        n_rnn      = config['n_rnn'],
        n_output   = N_OUTPUT,
        l2_h       = 1e-6,
        l2_weight  = 1e-6,
    )
    check('GRUNet created', True,
          f'{sum(p.numel() for p in gru.parameters()):,} params')

    gru.eval()
    with torch.no_grad():
        result_gru = gru(x)

    check('GRU output shape correct',
          result_gru.output.shape == (trial.tdim, config['batch_size'], N_OUTPUT),
          f'{tuple(result_gru.output.shape)}')
    check('GRU output values finite',
          torch.isfinite(result_gru.output).all().item())

except Exception as e:
    check('GRU forward pass', False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Loss and performance
# ---------------------------------------------------------------------------

section('Loss and Performance')

try:
    trial  = YANG_DRISCOLL_TASKS['delaypro'](config, rng)
    x, y, c = trial.to_tensors()

    model.eval()
    with torch.no_grad():
        result = model(x)

    loss = masked_mse_loss(result.output, y, c)
    check('masked_mse_loss runs',     True)
    check('loss is scalar',           loss.ndim == 0)
    check('loss is positive',         loss.item() > 0,    f'loss={loss.item():.4f}')
    check('loss is finite',           torch.isfinite(loss).item())

    perf = compute_performance(result.output, y, c)
    check('compute_performance runs', True)
    check('perf in [0, 1]',           0.0 <= perf <= 1.0, f'perf={perf:.3f}')

    # Untrained network should have low performance.
    check('untrained perf < 0.5',     perf < 0.5,
          f'perf={perf:.3f} (untrained network, expected near 0)')

except Exception as e:
    check('Loss/performance', False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------

section('One Training Step (end-to-end)')

try:
    import tempfile, os

    small_config = make_config(n_rnn=32, batch_size=4)
    small_model  = build_network({**small_config, 'rnn_type': 'LeakyRNN'})
    small_dataset = TaskDataset(
        task_funcs        = YANG_DRISCOLL_TASKS,
        config            = small_config,
        batches_per_epoch = 5,
        seed              = 1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model    = small_model,
            dataset  = small_dataset,
            config   = small_config,
            save_dir = tmpdir,
        )
        check('Trainer created', True)

        # Run 3 steps manually.
        from torch.utils.data import DataLoader
        loader = DataLoader(small_dataset, batch_size=1,
                            shuffle=False, collate_fn=collate_trials)
        loader_iter = iter(loader)
        losses = []
        for _ in range(3):
            x, y, c, _ = next(loader_iter)
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            c = c.to(trainer.device)
            loss, reg = trainer._train_step(x, y, c)
            losses.append(loss)

        check('3 training steps ran',   True)
        check('loss decreased or stable',
              losses[-1] <= losses[0] * 2.0,   # very loose — just checking it's not exploding
              f'losses: {[f"{l:.4f}" for l in losses]}')
        check('loss values finite',
              all(np.isfinite(l) for l in losses),
              f'losses: {losses}')

        # Save checkpoint.
        trainer.step = 3
        trainer.save_checkpoint('test_ckpt.pt')
        ckpt_path = os.path.join(tmpdir, 'test_ckpt.pt')
        check('checkpoint saved', os.path.exists(ckpt_path))

        # Load checkpoint.
        trainer2 = Trainer(
            model    = build_network({**small_config, 'rnn_type': 'LeakyRNN'}),
            dataset  = small_dataset,
            config   = small_config,
            save_dir = tmpdir,
        )
        trainer2.load_checkpoint(ckpt_path)
        check('checkpoint loaded', trainer2.step == 3)

except Exception as e:
    check('Training step', False, str(e))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# build_network factory
# ---------------------------------------------------------------------------

section('build_network factory')

for rnn_type in ['LeakyRNN', 'GRU']:
    try:
        net = build_network({**config, 'rnn_type': rnn_type})
        trial = YANG_DRISCOLL_TASKS['delaypro'](config, rng)
        x, _, _ = trial.to_tensors()
        net.eval()
        with torch.no_grad():
            out = net(x)
        check(f'build_network({rnn_type})', True,
              f'{sum(p.numel() for p in net.parameters()):,} params')
    except Exception as e:
        check(f'build_network({rnn_type})', False, str(e))

try:
    build_network({**config, 'rnn_type': 'invalid'})
    check('invalid rnn_type raises', False, 'should have raised ValueError')
except ValueError:
    check('invalid rnn_type raises ValueError', True)

section('New Task Generation (T12-T30)')

all_passed = True
for task_name, task_fn in NEW_TASKS.items():
    try:
        trial = task_fn(config, rng)
        ok_x      = trial.x.shape == (trial.tdim, config['batch_size'], N_INPUT)
        ok_y      = trial.y.shape == (trial.tdim, config['batch_size'], N_OUTPUT)
        ok_mask   = trial.c_mask.sum() > 0
        ok_epochs = isinstance(trial.epochs, dict) and len(trial.epochs) > 0
        ok_finite = np.isfinite(trial.x).all() and np.isfinite(trial.y).all()
        all_ok    = all([ok_x, ok_y, ok_mask, ok_epochs, ok_finite])
        detail    = f'T={trial.tdim} B={config["batch_size"]} epochs={list(trial.epochs.keys())}'
        check(f'{task_name:<22}', all_ok, detail)
        if not all_ok:
            all_passed = False
    except Exception as e:
        check(f'{task_name:<22}', False, str(e))
        traceback.print_exc()
        all_passed = False

print(f'\n  New tasks: {"all passed" if all_passed else "some failed"}')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f'\n{"="*60}')
print('  Test complete.')
print('  If all checks show PASS, the codebase is ready for training.')
print('  Fix any FAIL items before submitting a cluster job.')
print(f'{"="*60}\n')