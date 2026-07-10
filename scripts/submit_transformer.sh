#!/bin/bash
#SBATCH --job-name=meta_plastic_transformer_30tasks
#SBATCH --account=kempner_dgc_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_transformer_%j.out
#SBATCH --error=logs/train_transformer_%j.err
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
# Training
# Transformer hyperparameters:
#   d_model=128  internal embedding dimension
#   n_heads=4    attention heads (must divide d_model)
#   n_layers=3   transformer encoder layers
#   d_ff=256     feedforward layer dimension
#
# Note: n_rnn is unused for Transformer but required by argparse.
#       d_model controls the network size instead.
# ---------------------------------------------------------------------------

echo "Starting Transformer 30-task training at $(date)"

python3 -u scripts/train_all30.py \
    --rnn_type     Transformer \
    --n_rnn        128 \
    --d_model      128 \
    --n_heads      4 \
    --n_layers     3 \
    --d_ff         256 \
    --seed         0 \
    --max_steps    10000000 \
    --batch_size   64 \
    --lr           0.0001 \
    --display_step 5000 \
    --ckpt_step    50000 \
    --target_perf  1.1 \
    --task_subset  all30 \
    --save_dir     runs/Transformer_128d_30tasks_seed0

echo "Training complete at $(date)"