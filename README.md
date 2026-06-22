# DeepSecE

Fine-tuned model of combined approach detection of for inclusion membrane proteins and host-translocated type 3 effectors of Chlamydiae. We combine 2 pretrained models:
- DeepSecE: Secretion-specific Transformer model used in secretion protein prediction in Gram-negative bacteria (https://doi.org/10.34133/research.0258)
- TMbed: CCN-based model for per-residue prediction of transmembrane segments and topology (https://doi.org/10.1186/s12859-022-04873-x)

The combined model has been fine-tuned on a curated data base of chlamydia T3 effectors, resulting in an accurate classifier of chlamydial proteins in 3 classes:

- Inclusion membrane proteins (Incs) - T3 effectors inserted in the inclusion membrane
- Host-translocated effectors - T3 effectors located in the host cell
- Negatives - Non-T3 effectors

## Performance Comparison

## Set up

### Requirements (DeepSecE and TMbed requirements)

- python==3.9.7
- torch==1.10.2
- biopython==1.79
- einops==0.4.1
- fair-esm>=0.4.0
- tqdm==4.64.0
- numpy==1.21.2
- scikit-learn==0.23.2
- matplotlib==3.5.1
- seaborn==0.11.0
- tensorboardX==2.0
- umap-learn==0.5.3
- warmup-scheduler==0.3.2
- h5py >= "3.2.1"
- sentencepiece >= "0.1.96"
- transformers >= "4.11.3"
- typer >= "0.4.1"

### Installation

To install clone this repository and run directly:

```shell
git clone ""

```

If you want to plot the sequence attention, you should install package `logomarker`

```shell
pip install logomaker
```

## Usage

### Train model

Command used for model fine-tuning:

```shell
for i in {0..4}
do
python3 DeepSecE_for_ft/train.py --model effectortransformer \
--data_dir data/DeepsecE_ft/training_set_ft16 \
--batch_size 32 --lr 5e-5 \
--weight_decay 4e-5 \
--dropout_rate 0.4 \
--num_layers 1 \
--num_heads 4 \
--max_epochs 200 \
--warm_epochs 1 \
--patience 5 \
--lr_scheduler cosine \
--lr_decay_steps 30 \
--kfold 5 \
--fold_num $i \
--log_dir runs/attempt_cv \
--model_initial model/checkpoint.pt \
--with_tmbed 1 \
--mha 1
done
```

 Parameters:

- `--model` train a transformer or finetune a ESM-1b model.
- `--batch_size` 32 --lr 5e-5 \
- `--weight_decay` 
- `--dropout_rate` 
- `--max_epochs` 
- `--warm_epochs` 
- `--lr_decay_steps`
- `--model_initial`
- `--data_dir` directory that stores training data (default: ./data).
- `--num_layers` numbers of trainable transformer layer. (default: 1)
- `--num_heads` numbers of attention heads in secretion-specific transformer (default: 4).
- `--patience` patience for early stopping used in training.
- `--lr_schedular` learning rate schedular [step, consine].
- `--log_dir` directory that stores training outputs (default: logs).
- `--with_tmbed`
- `--mha 1`

### Prediction

Input: fasta file containing protein(s) or proteome of interest.

#### Command used for prediction:

```shell
python3 ~/source/DeepSecE_for_ft/predict.py \
					--fasta_path input_file.fasta \
					--model_location model/checkpoint.pt \
					--out_dir output_directory \
					--save_attn \
					--save_embedding \
					--save_umap \
```

Parameters:

- `--fasta_path` path to the input protein FASTA file.
- `--model_location` path to the model weights
- `--out_dir` directory that stores prediction outputs.
- `--save_attn` add to save sequence attention weights for DeepSecE and TMbed (need for attention logo plots).
- `--save_embedding` add to save sequence embeddings (TMbed+DeepSecE embedding, input for classification layer. Needed for UMAP plots)
- `--save_umap` add to save UMAP projection of sequence embeddings
- `--no_cuda` add when CUDA is not available.

Output that will be saved in `out_dir` includes:
- `predictions.csv` file with results of prediction: class predicted for each protein, probabilities assigned to each class, sequence length.
- `deepSecE_attn.npz` attention weights for DeepSecE, used for sequence attn logos (saved if save_attn = True)
- `tmbed_mha.npz`  attention weights for TMbed, used for sequence attn logos (saved if save_attn = True)
- `umap.png` UMAP projection of sequence embeddings (saved if save_umap = True)
- `effectors.fasta` fasta file containing all input sequences with predicted class in its annotation


It takes about ---- minutes to compute predictionns for a proteome size = with GPU ------- and --- with CPU

### Plot attention

If you save the attention output of the putative secreted proteins (add `--save_attn`), you can run `python scripts/plot_attention.py [directory of prediction output]` to plot the saliency map from attention, and infer potentially import regions related to protein secretion.

## Contact

Please contact Maria Colonna (maria.colonna@pasteur.fr) for any questions, comments or issues.
