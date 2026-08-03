#!/bin/bash
#SBATCH --job-name=transformer_256d_30tasks
#SBATCH --account=kempner_dgc_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_transformer256_%j.out
#SBATCH --error=logs/train_transformer256_%j.err
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
# Transformer d_model=256 — matched to LeakyRNN and GRU hidden size
#
# Purpose: clean four-way comparison (LeakyRNN, GRU, Transformer, PlasticRNN)
# with all recurrent/embedding dimensions equal at 256.
#
# Previously ran Transformer at d_model=128 which confounded the
# cross-architecture representational geometry comparison.
#
# Hyperparameters:
#   d_model=256   matches LeakyRNN/GRU n_rnn=256
#   n_heads=4     divides 256 evenly (head_dim=64)
#   n_layers=3    same as previous transformer
#   d_ff=512      2x d_model (standard ratio)
#   lr=1e-4       standard transformer lr
#   batch_size=64 reduced from 128 (transformer uses more memory)
# ---------------------------------------------------------------------------

echo "Starting Transformer d_model=256 training at $(date)"

python3 -u scripts/train_all30.py \
    --rnn_type     Transformer \
    --n_rnn        256 \
    --d_model      256 \
    --n_heads      4 \
    --n_layers     3 \
    --d_ff         512 \
    --seed         0 \
    --max_steps    10000000 \
    --batch_size   64 \
    --lr           0.0001 \
    --display_step 5000 \
    --ckpt_step    50000 \
    --target_perf  1.1 \
    --task_subset  all30 \
    --save_dir runs/Transformer_256d_30tasks_seed0_v2

echo "Training complete at $(date)"