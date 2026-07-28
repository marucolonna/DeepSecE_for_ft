#!/bin/bash
###################################################################################################################################

##################################################### FINE TUNING PREDINC #####################################################

###################################################################################################################################
##################################################### Maria Colonna ###############################################################
###################### Cellular Biology of Microbial Infection - Structural Bioinformatics, Institut Pasteur ######################
###################################### https://research.pasteur.fr/fr/member/maria-colonna/ #######################################
###################################################################################################################################

@echo "Will be run on maestro (Large GPU memory required)"

sshrun maestro.pasteur.fr "sbatch PredInc/scripts/PredInc_train.slurm.sh $@" \
							--remote-tmp-parent-dir /pasteur/helix/scratch/mcolonna/DeepSecE_ft/ft_18 \
							--force-keep \
							--transfer PredInc/scripts/PredInc_train.slurm.sh \
									PredInc/apptainer/PredInc.sif \
									data/train_set/training_data_ft18/labeled_train_set.fasta \
									PredInc/weights/DeepSecE/checkpoint.pt \
									PredInc/weights/tmbed/cnn \
							--remote-dir /pasteur/helix/scratch/mcolonna/DeepSecE_ft/ft_18
