"""Generate scGPT cell embeddings for an h5ad file."""
import argparse
from pathlib import Path

import anndata as ad
import scanpy as sc
import scgpt as scg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input h5ad file")
    parser.add_argument("--output", required=True, help="Output h5ad file")
    parser.add_argument("--model-dir", required=True, help="scGPT model directory")
    parser.add_argument("--obsm-key", default="X_scgpt", help="Key for embeddings in obsm")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading {args.input}")
    adata = sc.read_h5ad(args.input)
    print(f"Shape: {adata.shape}")

    # scGPT needs raw counts and gene_symbol column
    adata_embed = adata.copy()
    adata_embed.var["gene_symbol"] = adata_embed.var_names.values
    if "counts" in adata_embed.layers:
        adata_embed.X = adata_embed.layers["counts"]

    print(f"Running scGPT embedding with model from {args.model_dir}")
    result = scg.tasks.embed_data(
        adata_embed,
        args.model_dir,
        gene_col="gene_symbol",
        batch_size=args.batch_size,
        return_new_adata=True,
    )

    adata.obsm[args.obsm_key] = result.X
    print(f"Embeddings shape: {result.X.shape}, stored as '{args.obsm_key}'")

    print(f"Saving to {args.output}")
    adata.write_h5ad(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
