#!/bin/bash
###################################################################################################################################

##################################################### PREDICTION WITH PREDINC #####################################################

###################################################################################################################################
##################################################### Maria Colonna ###############################################################
###################### Cellular Biology of Microbial Infection - Structural Bioinformatics, Institut Pasteur ######################
###################################### https://research.pasteur.fr/fr/member/maria-colonna/ #######################################
###################################################################################################################################

#Prediction from input fasta (1 protein, several proteins or full proteome accepted)
sshrun maestro.pasteur.fr "sbatch PredInc/scripts/prediction.slurm.sh $1" \
 								--remote-tmp-parent-dir /pasteur/helix/scratch/mcolonna \
 								--force-keep \
 								--transfer PredInc/scripts/prediction.slurm.sh \
 										PredInc/predict.py \
 										PredInc/apptainer/PredInc.sif \
										PredInc/weights/DeepSecE/checkpoint.pt \
 										PredInc/weights/PredInc/predinc100_checkpoint.pt \
 										PredInc/weights/tmbed/cnn/cv_0.pt \
 										$1 \
 								--remote-dir /pasteur/helix/scratch/mcolonna/PredInc-predict

