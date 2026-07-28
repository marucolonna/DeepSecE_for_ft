#!/usr/bin/env python3
#
# This file is a derivative work based on [Project DeepSecE],
# originally licensed under the [MIT License].
# Original source: [https://github.com/zhangyumeng1sjtu/DeepSecE/tree/main]
# Copyright (c) 2022 Yumeng Zhang

# Modifications made by Maria Colonna, Institut Pasteur, 2026,
# are licensed under the [XXX License].

import os
from pathlib import Path
import random
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from esm import Alphabet, FastaBatchedDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from DeepSecE.model import EffectorTransformer
from scripts.umap import plot_umap
from scripts.plot_mha import plot_mha
from scripts.plot_attention import plot_attention


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def predict(model, fasta, batch_size, device, outdir, pos_labels, save_attn=False, save_embedding=False):
    predicted_labels = ['Negative', 'Inc-protein', 'Secreted effector'] #incfold
    print(f'Loading FASTA Dataset from {fasta}')

    dataset = FastaBatchedDataset.from_file(fasta)
    alphabet = Alphabet.from_architecture("roberta_large")
    loader = DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), num_workers=4, batch_size=batch_size)

    model.eval()
    probs = []
    preds = []
    names = []
    lengths = []
    seq_records = []

    if save_attn:
        attn_dict = {}
        mha_dict = {}

    with torch.no_grad():
        embeddings = []
        protein_names = []
        for labels, strs, toks in tqdm(loader):
            toks = toks.to(device)
            if save_attn:
                if save_embedding:
                    out, embedding, attn, mha_weights = model(strs, toks) #embedding [1, 244]
                    attn = attn.cpu().numpy()
                    mha_weights = mha_weights.squeeze().cpu().numpy()
                    
                else:
                    out, attn, mha_weights = model(strs, toks)
                    attn = attn.cpu().numpy()
                    mha_weights = mha_weights.squeeze().cpu().numpy()

            else:
                if save_embedding:
                    out, embedding = model(strs, toks)
                else:   
                    out = model(strs, toks)
            
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(prob, 1)

            pred = torch.zeros(prob.shape[0], dtype=torch.long)  # Start with all Negative (0)
            pred[prob[:, 1] >= 0.8] = 1  #Inc-protein (1) if prob >= 0.8
            #pred[prob[:, 2] >= 0.8] = 2 #Secreted effector (2) if prob >= 0.8 #removing secreted effector class, only Incs or Negative
           
            probs.append(prob.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            
            protein_name = labels[0].split()[0] #only working for batch size=1
        
            protein_names.append(protein_name)
            
            if save_embedding:
                embeddings.append(embedding.detach().cpu())
              
            if save_attn:
                mha_dict[protein_name] = mha_weights

            for i, str in enumerate(strs):
                name = labels[i].split()[0]
                pred_label = predicted_labels[pred[i].cpu().numpy()]
                if pred_label in pos_labels:
                    if save_attn:
                        seq = str[:1020]
                        avg_attn = attn[i, :, :len(seq), :len(seq)].sum(0).mean(0)
                        attn_dict[name] = avg_attn
                        
                    record = SeqRecord(Seq(str), id=name, description=f'predicted class: {pred_label} ')
                    seq_records.append(record)
                names.append(name)
                lengths.append(len(str))
    
        protein_names_df = pd.DataFrame({'name': protein_names})
        protein_names_df.to_csv(os.path.join(outdir, 'seq_labels.csv'), index=False)

        if save_embedding:
            embeddings = torch.cat(embeddings, dim=0)  # [N, 244] for umap incfold
            torch.save(embeddings, os.path.join(outdir, "seq_embeddings.pt")) #incfold - save embeddings for Ft

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    print(f"{probs.shape=}")  # all sequences !

    probs_nega= probs[:, 0] #incfold
    probs_inc= probs[:, 1] #incfold
    probs_sec= probs[:, 2] #incfold

    pred_class = list(map(lambda x: predicted_labels[x], preds)) #incfold
    scores = [prob[idx] for prob, idx in zip(probs, preds)]

    result = pd.DataFrame({'name': names, 'pred_class': pred_class, 'score': scores, 'inc.prob': probs_inc, 'nega.prob': probs_nega, 'sec.prob': probs_sec, 'length': lengths}) #incfold
    result = result.round(4)
    print(f"{result.shape=}")  # all sequences !

    print(f"Writing prediction result in {os.path.join(outdir, 'predictions.csv')}")
    result.to_csv(os.path.join(outdir, 'predictions.csv'), index=False)

    print(f"Writing putative inc proteins in {os.path.join(outdir, 'effectors.fasta')}") #incfold
    SeqIO.write(seq_records, os.path.join(outdir, 'effectors.fasta'), 'fasta') #incfold

    if save_attn:
        print(f"Saving inc protein attention in {os.path.join(outdir, 'DeepSecE_attn.npz')} and {os.path.join(outdir, 'TMbed_attn.npz')}") #incfold
        np.savez(os.path.join(outdir, 'DeepSecE_attn.npz'), **attn_dict)
        np.savez(os.path.join(outdir, 'TMbed_attn.npz'), **mha_dict)

def main(args):

    set_seed(42)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Using device {device} for prediction')
    start_time = time.time()

    model = EffectorTransformer(1280, 33, hid_dim=256, num_layers=1, heads=4,
                            dropout_rate=0.4, num_classes=3, return_embedding=args.save_embedding, return_attn=args.save_attn)
    model.to(device)
    
    print(f'Loading model from {args.model_location}')
    if args.no_cuda:
        model_weights = torch.load(args.model_location, map_location="cpu")

        tmbed_weights_file = Path('PredInc/weights/tmbed/cnn/cv_0.pt') #incfold - tmbed weights
        tmbed_weights = torch.load(tmbed_weights_file, map_location="cpu")
        tmbed_weights = tmbed_weights['model']
        tmbed_weights = {'tmbed.' + k: v for k, v in tmbed_weights.items()} #incfold

        model_weights = {**model_weights, **tmbed_weights}
        
        model.load_state_dict(model_weights)
     
    else:
        model_weights = torch.load(args.model_location)
        
        tmbed_weights_file = Path('PredInc/weights/tmbed/cnn/cv_0.pt') #incfold - tmbed weights
        tmbed_weights = torch.load(tmbed_weights_file)
        tmbed_weights = tmbed_weights['model']
        tmbed_weights = {'tmbed.' + k: v for k, v in tmbed_weights.items()} #incfold

        model_weights = {**model_weights, **tmbed_weights}

        model.load_state_dict(model_weights)

    predict(model, args.fasta_path, args.batch_size, device,
            args.out_dir, args.labels, args.save_attn, args.save_embedding)

    end_time = time.time()
    secs = end_time - start_time

    print(f'It took {secs:.1f}s to finish the prediction')

    if args.save_attn:
        plot_mha(os.path.join(args.out_dir, 'TMbed_attn.npz'), args.fasta_path) #saves attn profiles for TMbed
        plot_attention(os.path.join(args.out_dir, 'DeepSecE_attn.npz'), args.fasta_path) #saves attn profiles for DeepSecE

    if args.save_embedding:
        seq_embeddings = os.path.join(args.out_dir, "seq_embeddings.pt")
        seq_names = os.path.join(args.out_dir, "protein_names_embeddings.csv")
        seq_predictions = os.path.join(args.out_dir, "predictions.csv")
        output = os.path.join(args.out_dir, "umap.png")
        plot_umap(seq_embeddings, seq_names, seq_predictions, output) #plots and saves figure

if __name__ == '__main__':

    parser = ArgumentParser(
        description="Predict secreted substrate proteins from protein sequences in a FASTA file.")
    
    parser.add_argument('--batch_size', default=1, type=int,
                        help='bacth size used in prediction. (default: 1)')
    parser.add_argument('--fasta_path', required=True, type=str,
                        help='input ordered protein sequences.')
    parser.add_argument('--model_location', required=True, type=str,
                        help='path to the model weights.')
    parser.add_argument('--labels', nargs='+', default=['Negative', 'Inc-protein', 'Secreted effector'],
                        help='types of secreted proteins requiring prediction. (default: Inc_, nega, secreted)') #incfold
    parser.add_argument('--out_dir', default='./', type=str,
                        help='output directory of prediction results.')
    parser.add_argument('--save_attn', action='store_true',
                        help='save the sequence attention of inc proteins.') #incfold
    parser.add_argument('--no_cuda', action='store_true',
                        help='add when CUDA is not available.')
    parser.add_argument('--save_embedding', action='store_true',
                        help='save the sequence embedding of inc proteins for FT.') #incfold
    parser.add_argument('--umap', action='store_true',
                        help='add if you want to obtain a UMAP ofyour sequence embeddings - only possible if embeddings are saved') #incfold
    parser.add_argument('--attn_profile', action='store_true',
                        help='add if you want to obtain a profiles for the attn weights of DeepSecE and TMbed - only possible if attn is saved') #incfold
    args = parser.parse_args()

    main(args)
