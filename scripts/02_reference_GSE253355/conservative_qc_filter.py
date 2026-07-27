import scanpy as sc
from pathlib import Path

IN = Path("/Spatial_Therapy_OU_Levy_Branching/GSE253355/processed/GSE253355_normal_BM_log_normalized.h5ad")
OUT = IN.parent / "GSE253355_normal_BM_reference_filtered.h5ad"

adata = sc.read_h5ad(IN)

print("Before:", adata)

# Conservative QC filter
adata = adata[
    (adata.obs["n_genes_by_counts"] >= 300) &
    (adata.obs["n_genes_by_counts"] <= 9000) &
    (adata.obs["pct_counts_mt"] <= 20)
].copy()

print("After:", adata)

adata.write_h5ad(OUT)
print("Saved:", OUT)
