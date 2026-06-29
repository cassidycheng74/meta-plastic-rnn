#!/bin/bash
#SBATCH --job-name=meta_plastic_rnn_30tasks
#SBATCH --account=kempner_dgc_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/train_all30_%j.out
#SBATCH --error=logs/train_all30_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ccheng35@harvard.edu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

cd ~/projects/meta-plastic-rnn

# Create logs directory if it doesn't exist.
mkdir -p logs

# Load CUDA modules.
module purge
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01

# Activate venv.
source .venv/bin/activate

# Confirm GPU is visible.
echo "=== GPU Info ==="
nvidia-smi
echo "=== Python/Torch ==="
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo "================"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

echo "Starting 30-task training at $(date)"

python3 scripts/train_all30.py \
    --rnn_type    LeakyRNN \
    --n_rnn       128 \
    --seed        0 \
    --max_steps   10000000 \
    --batch_size  64 \
    --lr          0.001 \
    --display_step 1000 \
    --ckpt_step   10000 \
    --target_perf 0.99 \
    --task_subset all30

echo "Training complete at $(date)"
