#!/bin/bash
#SBATCH --job-name=plastic_toggle_only
#SBATCH --account=kempner_dgc_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_plastic_toggle_%j.out
#SBATCH --error=logs/train_plastic_toggle_%j.err
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

echo "Starting toggle-only PlasticRNN training at $(date)"

python3 -u scripts/train_plastic.py \
    --n_rnn         128 \
    --seed          0 \
    --n_lifetimes   100000 \
    --n_trials      20 \
    --batch_size    8 \
    --lr            0.0003 \
    --alpha_init    0.0 \
    --eta_init      0.01 \
    --l2_alpha      0.0001 \
    --hebb_clip     0.5 \
    --w_rec_coeff   0.5 \
    --display_every 500 \
    --ckpt_every    5000 \
    --task_subset   toggle_only \
    --save_dir      runs/PlasticRNN_128units_toggle_T20_seed0

echo "Training complete at $(date)"