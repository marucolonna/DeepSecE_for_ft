#!/usr/bin/env python3
import os
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


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def predict(model, fasta, batch_size, device, outdir, pos_labels, save_attn=False):
    predicted_labels = ['Inc-protein', 'Negative'] #incfold
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

    with torch.no_grad():
        for labels, strs, toks in tqdm(loader):
            toks = toks.to(device)
            if save_attn:
                out, attn = model(strs, toks)
                attn = attn.cpu().numpy()
            else:
                out = model(strs, toks)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(prob, 1)

            probs.append(prob.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

            for i, str in enumerate(strs):
                name = labels[i].split()[0]
                pred_label = predicted_labels[pred[i].cpu().numpy()]
                if pred_label in pos_labels:
                    if save_attn:
                        seq = str[:1020]
                        avg_attn = attn[i, :, :len(seq), :len(seq)].sum(0).mean(0)
                        attn_dict[name] = avg_attn
                    record = SeqRecord(Seq(str), id=name, description=f'putative type {pred_label} secreted protein')
                    seq_records.append(record)
                names.append(name)
                lengths.append(len(str))

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    print(f"{probs.shape=}")  # all sequences !

    probs_inc= probs[:, 0] #incfold
    probs_nega= probs[:, 1] #incfold

    is_inc = list(map(lambda x: predicted_labels[x], preds)) #incfold
    scores = [prob[idx] for prob, idx in zip(probs, preds)]

    result = pd.DataFrame({'name': names, 'is_inc': is_inc, 'score': scores, 'inc.prob': probs_inc, 'nega.prob': probs_nega, 'length': lengths}) #incfold
    result = result.round(4)
    print(f"{result.shape=}")  # all sequences !

    print(f"Writing prediction result in {os.path.join(outdir, 'predictions.csv')}")
    result.to_csv(os.path.join(outdir, 'predictions.csv'), index=False)

    effector = result[result['is_inc'].isin(pos_labels)] #incfold
    print(f"{effector.shape=}")
    effector.to_csv(os.path.join(outdir, 'results.csv'), index=False)

    print(f"Writing putative inc proteins in {os.path.join(outdir, 'inc_proteins.fasta')}") #incfold
    SeqIO.write(seq_records, os.path.join(outdir, 'inc_proteins.fasta'), 'fasta') #incfold

    if save_attn:
        print(f"Saving inc protein attention in {os.path.join(outdir, 'attn.npz')}") #incfold
        np.savez(os.path.join(outdir, 'attn.npz'), **attn_dict)


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
                            dropout_rate=0.4, num_classes=2, return_attn=args.save_attn)
    model.to(device)
    
    print(f'Loading model from {args.model_location}')
    if args.no_cuda:
        model.load_state_dict(torch.load(args.model_location, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(args.model_location))

    predict(model, args.fasta_path, args.batch_size, device,
            args.out_dir, args.is_inc_labels, args.save_attn)

    end_time = time.time()
    secs = end_time - start_time

    print(f'It took {secs:.1f}s to finish the prediction')


if __name__ == '__main__':

    parser = ArgumentParser(
        description="Predict secreted substrate proteins from protein sequences in a FASTA file.")
    
    parser.add_argument('--batch_size', default=1, type=int,
                        help='bacth size used in prediction. (default: 1)')
    parser.add_argument('--fasta_path', required=True, type=str,
                        help='input ordered protein sequences.')
    parser.add_argument('--model_location', required=True, type=str,
                        help='path to the model weights.')
    parser.add_argument('--is_inc_labels', nargs='+', default=['Inc_', 'nega'],
                        help='types of secreted proteins requiring prediction. (default: Inc_, nega)') #incfold
    parser.add_argument('--out_dir', default='./', type=str,
                        help='output directory of prediction results.')
    parser.add_argument('--save_attn', action='store_true',
                        help='save the sequence attention of inc proteins.') #incfold
    parser.add_argument('--no_cuda', action='store_true',
                        help='add when CUDA is not available.')

    args = parser.parse_args()

    main(args)
