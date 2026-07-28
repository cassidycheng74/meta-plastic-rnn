#!/bin/bash
#SBATCH --job-name=leakyrnn_256_30tasks_v4
#SBATCH --account=kempner_dgc_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_all30_%j.out
#SBATCH --error=logs/train_all30_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccheng35@harvard.edu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

cd ~/projects/meta-plastic-rnn
mkdir -p logs

module purge
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01

source .venv/bin/activate

echo "=== GPU Info ==="
nvidia-smi
echo "=== Python/Torch ==="
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"
echo "================"

# ---------------------------------------------------------------------------
# Training — v4 with fixed task definitions
#
# Changes from v3:
#   - T16 RhythmGeneration: ramp-up at start of sustain
#   - T18 Toggle: initial state signaled on IN_REAL_B
#   - T19 ConditionalToggle: initial states signaled on IN_REAL_B, IN_SIN_B
#   - T28 ConditionalRhythm: phase continuity fix at switch
# ---------------------------------------------------------------------------

echo "Starting 30-task training v4 (fixed tasks) at $(date)"

python3 -u scripts/train_all30.py \
    --rnn_type     LeakyRNN \
    --n_rnn        256 \
    --seed         0 \
    --max_steps    10000000 \
    --batch_size   128 \
    --lr           0.0003 \
    --display_step 5000 \
    --ckpt_step    50000 \
    --target_perf  1.1 \
    --task_subset  all30 \
    --save_dir     runs/LeakyRNN_256units_30tasks_seed0_v4

echo "Training complete at $(date)"