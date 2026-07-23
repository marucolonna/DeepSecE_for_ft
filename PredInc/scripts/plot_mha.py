import sys
import numpy as np
import matplotlib.pyplot as plt
import logomaker as lm
from Bio import SeqIO
import pandas as pd
import os



def plot_mha(mha_file, fasta_file):
    data = np.load(mha_file) #mha.npz

    for key, value in data.items():
        mha_matrix = value[:-1] #remove eos position (is there because of pT5 tokenization)
        prot_sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

        seq= None
        prot_name = mha_file.split("/")[-1].split(".npy")[0]

        for key, value in prot_sequences.items():
            if  prot_name in key:
                seq = str(value.seq)
                print(f"Name: {key}, Sequence Length: {len(seq)}")
                print(f"Sequence Length: {len(seq)}, MHA Matrix Length: {len(mha_matrix)}")
            else:
                continue

        if seq != None:
            alphabet = sorted(set(seq))
            data = []
            for res, w in zip(seq, mha_matrix):
                row = {aa: 0 for aa in alphabet}
                row[res] = w
                data.append(row)

            logo_df = pd.DataFrame(data)
            logo = lm.Logo(logo_df, color_scheme="hydrophobicity")

            logo.style_spines(visible=False)
            logo.style_spines(spines=['left', 'bottom'], visible=True)
            logo.ax.set_ylabel('Weight')
            logo.ax.set_xlabel('Position')
            logo.ax.set_title('Average Multi-Head Attention Weights per Residue Position')

            # Tight layout for saving
            plt.tight_layout()
            #logo.ax.figure.savefig(os.path.join(out_dir, mha_file.split("/")[-1].replace(".npy", ".png")), dpi=300)
            logo.ax.figure.savefig(f"TMbed_mha_logo/{prot_name}.png", dpi=300)
            #print("mha matrix shape:", mha_matrix.shape)
            #print("Min / Max in mha matrix:", mha_matrix.min(), "/", mha_matrix.max())
            print(f"Saved attention logo for {prot_name}")

#plot_attn(mha_file, fasta_file)

#mha_file = sys.argv[1]
#fasta_file = sys.argv[2]
#out_dir  = sys.argv[3]
#mha_matrix = np.load(mha_file, allow_pickle=True)
