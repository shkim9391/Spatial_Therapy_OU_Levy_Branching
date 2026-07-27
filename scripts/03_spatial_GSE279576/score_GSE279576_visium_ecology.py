from pathlib import Path
import scanpy as sc
import pandas as pd

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
IN = BASE / "processed"
OUTDIR = BASE / "processed" / "ecology_scores"
OUTDIR.mkdir(parents=True, exist_ok=True)

SIGNATURES = {
    "HSPC_primitive": ["CD34", "PROM1", "AVP", "MEIS1", "HOXA9", "KIT", "GATA2", "MLLT3"],
    "Myeloid_granulocytic": ["MPO", "ELANE", "AZU1", "PRTN3", "CTSG", "LTF", "S100A8", "S100A9"],
    "Monocyte_macrophage": ["LYZ", "LST1", "S100A8", "S100A9", "FCN1", "CTSS", "TYROBP", "MS4A7"],
    "Erythroid": ["HBB", "HBA1", "HBA2", "ALAS2", "KLF1", "GYPA", "AHSP"],
    "B_cell": ["MS4A1", "CD79A", "CD79B", "BANK1", "CD74", "IGHM"],
    "T_NK": ["CD3D", "CD3E", "TRAC", "NKG7", "GNLY", "GZMB", "IL7R"],
    "MSC_stromal": ["CXCL12", "LEPR", "PDGFRA", "PDGFRB", "VCAM1", "KITLG", "NT5E", "ENG"],
    "Endothelial_vascular": ["PECAM1", "VWF", "KDR", "ESAM", "EMCN", "CLDN5", "FLT1"],
    "Osteolineage_endosteal": ["BGLAP", "SPP1", "RUNX2", "COL1A1", "COL1A2", "IBSP", "ALPL"],
    "Adipocytic": ["ADIPOQ", "PLIN1", "FABP4", "LPL", "PPARG", "CEBPA"],
    "ECM_matrix": ["COL1A1", "COL1A2", "COL3A1", "FN1", "DCN", "LUM", "MMP2"],
    "Inflammatory": ["IL1B", "TNF", "CXCL8", "CCL2", "CCL3", "CCL4", "NFKBIA"],
    "CXCL12_CXCR4_axis": ["CXCL12", "CXCR4", "ACKR3", "KITLG", "VCAM1"],
    "Hypoxia_stress": ["HIF1A", "VEGFA", "SLC2A1", "LDHA", "BNIP3", "DDIT4"],

    # AML/spatial leukemia-focused add-ons
    "AML_blast_like": ["MPO", "AZU1", "PRTN3", "ELANE", "KIT", "CD34", "PROM1", "HOXA9"],
    "Primitive_like_AML": ["CD34", "PROM1", "KIT", "MEIS1", "HOXA9", "GATA2"],
    "Committed_like_AML": ["MPO", "AZU1", "ELANE", "PRTN3", "FCGR3B", "LTF"],
    "Immune_suppression": ["LILRB4", "LGALS9", "CD274", "HAVCR2", "TIGIT", "CTLA4"],
}

all_presence = []
all_sample_summaries = []

for f in sorted(IN.glob("*_visium_processed.h5ad")):
    sid = f.name.replace("_visium_processed.h5ad", "")
    print(f"\nScoring {sid}")

    adata = sc.read_h5ad(f)
    adata.var_names_make_unique()

    for name, genes in SIGNATURES.items():
        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]

        all_presence.append({
            "sample_id": sid,
            "signature": name,
            "n_genes_defined": len(genes),
            "n_genes_present": len(present),
            "genes_present": ",".join(present),
            "genes_missing": ",".join(missing),
        })

        if len(present) >= 2:
            sc.tl.score_genes(
                adata,
                gene_list=present,
                score_name=f"{name}_score",
                use_raw=False
            )
        else:
            adata.obs[f"{name}_score"] = 0.0

    score_cols = [c for c in adata.obs.columns if c.endswith("_score")]

    spot_scores = adata.obs[
        ["sample_id", "site", "version", "array_row", "array_col",
         "total_counts", "n_genes_by_counts", "pct_counts_mt"] + score_cols
    ].copy()

    # Add spatial coordinates
    spot_scores["spatial_x"] = adata.obsm["spatial"][:, 0]
    spot_scores["spatial_y"] = adata.obsm["spatial"][:, 1]

    spot_scores.to_csv(OUTDIR / f"{sid}_visium_ecology_scores_by_spot.csv", index=True)

    sample_summary = spot_scores[score_cols].agg(["mean", "std", "median"]).T
    sample_summary["sample_id"] = sid
    sample_summary["site"] = adata.obs["site"].iloc[0]
    sample_summary["version"] = adata.obs["version"].iloc[0]
    all_sample_summaries.append(sample_summary.reset_index().rename(columns={"index": "signature_score"}))

    out_h5ad = OUTDIR / f"{sid}_visium_ecology_scored.h5ad"
    adata.write_h5ad(out_h5ad)
    print("Saved:", out_h5ad)

presence = pd.DataFrame(all_presence)
presence.to_csv(OUTDIR / "GSE279576_visium_signature_gene_presence.csv", index=False)

summary = pd.concat(all_sample_summaries, ignore_index=True)
summary.to_csv(OUTDIR / "GSE279576_visium_ecology_scores_by_sample.csv", index=False)

print("\nSaved:")
print(OUTDIR / "GSE279576_visium_signature_gene_presence.csv")
print(OUTDIR / "GSE279576_visium_ecology_scores_by_sample.csv")
print("Done.")
