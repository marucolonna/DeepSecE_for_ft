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

export MY_SCRATCH_DIR="tmp"
export CACHE_DIR="${MY_SCRATCH_DIR}/.cache"
export CONFIG_DIR="${MY_SCRATCH_DIR}/.config"
export TORCH_HOME="$CACHE_DIR"
export MPLCONFIGDIR="$CONFIG_DIR"
export HF_HOME="$CACHE_DIR"  # Hugging Face cache    

input=$1 #fasta file or directory containing fastas
name=$(basename "${input}" .fasta)

if [ -f "$input" ]; then
	apptainer exec --nv PredInc/apptainer/PredInc.sif python3 /opt/PredInc/PredInc/predict.py \
					--fasta_path ${input} \
					--model_location PredInc/weights/PredInc/predinc100_checkpoint.pt \
					--out_dir outputs/"${name}"_results \
					--save_attn \
					--save_embedding
elif [ -d "$input" ]; then
	mkdir -p "$input"_results
	for FILE in "$input"/*; do
		name=$(basename "${FILE}" .fasta)
		echo "Processing $FILE"
		apptainer exec --nv PredInc/apptainer/PredInc.sif python3 /opt/PredInc/PredInc/predict.py \
					--fasta_path ${FILE} \
					--model_location PredInc/weights/PredInc/predinc100_checkpoint.pt \
					--out_dir outputs/"${name}"_results \
					--save_attn \
					--save_embedding
	done
else
    echo "Error: '$input' is not a valid file or directory"
    exit 1
fi
