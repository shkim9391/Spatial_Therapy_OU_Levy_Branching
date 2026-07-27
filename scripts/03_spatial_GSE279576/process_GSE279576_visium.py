from pathlib import Path
import scanpy as sc
import pandas as pd

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
RAW = BASE / "GSE279576_RAW"
OUT = BASE / "processed"
OUT.mkdir(exist_ok=True)

sample_ids = [x.strip() for x in (RAW / "sample_ids.txt").read_text().splitlines() if x.strip()]

for sid in sample_ids:
    print(f"\nProcessing {sid}")

    count_file = RAW / f"{sid}_filtered_feature_bc_matrix.h5"
    spatial_dir = RAW / "spatial_extracted" / sid

    adata = sc.read_visium(
        path=str(spatial_dir.parent / sid),
        count_file=str(count_file),
        source_image_path=str(spatial_dir / "spatial" / "tissue_hires_image.png")
    )

    adata.var_names_make_unique()
    adata.obs["sample_id"] = sid
    adata.obs["site"] = "BM" if "_BM" in sid else "EM"
    adata.obs["version"] = "v1" if sid.endswith("_v1") else "v2"

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # keep raw counts
    adata.layers["counts"] = adata.X.copy()

    # normalize for scoring / visualization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    out = OUT / f"{sid}_visium_processed.h5ad"
    adata.write_h5ad(out)

    qc = adata.obs[
        ["sample_id", "site", "version", "total_counts", "n_genes_by_counts", "pct_counts_mt"]
    ].describe(include="all")
    qc.to_csv(OUT / f"{sid}_qc_summary.csv")

    print(adata)
    print("Saved:", out)

print("\nDone.")
