#### UMAP dimensionality reduction of sequence embeddings from DeepSecE finetuning on Chlamydia trachomatis proteome (ft12) ####
# Maria Colonna, 2024-02-26 #

import umap
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_umap(seq_embeddings, seq_names, seq_predictions, output):

    embeddings = torch.load(seq_embeddings, map_location="cpu")

    embeddings_np = embeddings.detach().cpu().numpy()

    reducer = umap.UMAP(random_state=42)

    embedding_2d = reducer.fit_transform(embeddings_np)

    df = pd.DataFrame(embedding_2d, columns=["UMAP1", "UMAP2"])
    df["name"] = pd.read_csv(seq_names)['name'].values
    df["predicted_inc"] = pd.read_csv(seq_predictions)['pred_class'].values
    df["inc.prob"] = pd.read_csv(seq_predictions)['inc.prob'].values
    df["sec.prob"] = pd.read_csv(seq_predictions)['sec.prob'].values

    pd.DataFrame.to_csv(df, f"{output}_umap_values.csv", index=False)

    #Highlight validated Incs
    #highlights = pd.read_csv(args.highlights, names=['name'])
    #prots_to_highlight = np.array([1 if name in highlights['name'].values else 0 for name in df["name"]])

    # PLOT #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    sc = ax1.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=df["inc.prob"],
        cmap="viridis",        
        s=5
    )
    ax1.set_aspect('equal', 'datalim')
    cbar = fig.colorbar(sc, ax=ax1)
    cbar.set_label("Predicted P(Inc)", fontsize=8)
    ax1.set_title('UMAP projection of sequence embeddings', fontsize=10)

    #highlight_mask = prots_to_highlight.astype(bool)
    #if highlight_mask.any():
    #    ax1.scatter(
    #        embedding_2d[highlight_mask, 0],
    #        embedding_2d[highlight_mask, 1],
    #        c='yellow',
    #        s=20,   
    #        edgecolors='red',
    #        linewidths=0.2,
    #        alpha=0.8,
    #        label='Validated Incs'
    #    )

    sc2 = ax2.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=df["sec.prob"],
        cmap="plasma",        
        s=5
    )
    ax2.set_aspect('equal', 'datalim')
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("Predicted P(Secreted effector)", fontsize=8)
    ax2.set_title('UMAP projection of sequence embeddings', fontsize=10)

    plt.savefig(output, dpi=1200)
    plt.show()
