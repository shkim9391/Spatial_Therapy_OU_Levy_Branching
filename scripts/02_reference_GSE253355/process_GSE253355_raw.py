from pathlib import Path
import scanpy as sc
import anndata as ad
import pandas as pd

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE253355/GSE253355_RAW")
OUT = BASE.parent / "processed"
OUT.mkdir(exist_ok=True)

samples = sorted({p.name.replace("_matrix.mtx.gz", "") for p in BASE.glob("*_matrix.mtx.gz")})

adatas = []

for sid in samples:
    print(f"Processing {sid}")

    adata = sc.read_10x_mtx(
        BASE,
        prefix=f"{sid}_",
        var_names="gene_symbols",
        cache=False
    )

    adata.var_names_make_unique()
    adata.obs["sample_id"] = sid

    # Basic QC
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Conservative filtering for reference construction
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adatas.append(adata)

combined = ad.concat(
    adatas,
    join="outer",
    label="sample_id_batch",
    keys=samples,
    index_unique="-"
)

combined.var_names_make_unique()

print(combined)

# Save raw-count merged object
combined.write_h5ad(OUT / "GSE253355_normal_BM_raw_merged.h5ad")

# Normalize for exploratory reference analysis
adata = combined.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    flavor="seurat_v3",
    batch_key="sample_id",
    subset=False
)

adata.write_h5ad(OUT / "GSE253355_normal_BM_log_normalized.h5ad")

print("Saved:")
print(OUT / "GSE253355_normal_BM_raw_merged.h5ad")
print(OUT / "GSE253355_normal_BM_log_normalized.h5ad")
