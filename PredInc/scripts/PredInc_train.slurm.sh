#!/bin/bash
#SBATCH -p gpu       # Partition name (gpu)
#SBATCH --qos=gpu  # Quality of service (gpu)
# #BATCH --gres=gpu:1       # Number of GPUs (1)
# #SBATCH --mem-per-gpu=80G  # Request at least 48GB of GPU memory
#SBATCH --gres=gpu:1,gmem:80G # Request 80G
#SBATCH --cpus-per-task=8  # Number of CPU cores (8)
# #SBATCH --mem=<memory>       # Memory (16GB)
#SBATCH -N 1       # Number of nodes (1)

module load apptainer

export MY_SCRATCH_DIR="tmp"  # Use your actual scratch path!
export CACHE_DIR="${MY_SCRATCH_DIR}/.cache"
export CONFIG_DIR="${MY_SCRATCH_DIR}/.config"
export TORCH_HOME="$CACHE_DIR"          # PyTorch Hub models
export MPLCONFIGDIR="$CONFIG_DIR"
export HF_HOME="$CACHE_DIR"  # Hugging Face cache

for i in {0..1}
do
apptainer exec -B /pasteur/helix/scratch/mcolonna --nv PredInc/apptainer/PredInc.sif python3 /opt/PredInc/PredInc/train.py --model effectortransformer --data_dir data/train_set/training_data_ft18 --batch_size 32 --lr 5e-5 --weight_decay 4e-5 --dropout_rate 0.4 --num_layers 1 --num_heads 4 --max_epochs 200 --warm_epochs 1 --patience 5 --lr_scheduler cosine --lr_decay_steps 30 --kfold 5 --fold_num $i --log_dir runs/attempt_cv --model_initial PredInc/weights/DeepSecE/checkpoint.pt
done
