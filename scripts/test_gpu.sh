#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --account=kempner_dgc_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/test_gpu_%j.out

cd ~/projects/meta-plastic-rnn
module purge
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01
source .venv/bin/activate

echo "PyTorch version:"
python3 -c "import torch; print(torch.__version__)"

echo "CUDA available:"
python3 -c "import torch; print(torch.cuda.is_available())"

echo "GPU name:"
python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

echo "Running 10 training steps..."
python3 scripts/train_all30.py \
    --max_steps 100 \
    --display_step 50 \
    --batch_size 8 \
    --task_subset yang11
