"""Create a gene-subsetted version of op3_processed.h5ad using the gene list
from de_test.h5ad.

Reads:
  notebooks/neurips2025/perturbench_data/op3_processed.h5ad
  notebooks/neurips2025/perturbench_data/de_test.h5ad

Writes:
  notebooks/neurips2025/perturbench_data/op3_de_genes.h5ad
"""

from pathlib import Path

import scanpy as sc

DATA_DIR = Path(__file__).resolve().parent.parent / "notebooks/neurips2025/perturbench_data"
SRC = DATA_DIR / "op3_processed.h5ad"
GENE_REF = DATA_DIR / "de_test.h5ad"
OUT = DATA_DIR / "op3_de_genes.h5ad"


def main() -> None:
    print(f"Loading gene list from {GENE_REF}")
    de = sc.read_h5ad(GENE_REF, backed="r")
    gene_list = list(de.var_names)
    de.file.close()
    print(f"  {len(gene_list)} target genes")

    print(f"Loading {SRC}")
    adata = sc.read_h5ad(SRC)
    print(f"  shape: {adata.shape}")

    missing = [g for g in gene_list if g not in adata.var_names]
    if missing:
        raise ValueError(
            f"{len(missing)} target genes not in op3 (first 5: {missing[:5]})"
        )

    print("Subsetting to target genes (preserving de_test order)")
    adata = adata[:, gene_list].copy()
    print(f"  new shape: {adata.shape}")

    print(f"Writing {OUT}")
    adata.write_h5ad(OUT, compression="gzip")
    print("Done")


if __name__ == "__main__":
    main()
