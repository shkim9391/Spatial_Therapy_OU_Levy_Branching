from pathlib import Path
import scanpy as sc
import pandas as pd

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE253355")
IN = BASE / "processed" / "GSE253355_normal_BM_reference_filtered.h5ad"
OUTDIR = BASE / "processed" / "ecology_scores"
OUTDIR.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad(IN)
adata.var_names_make_unique()

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
}

present_summary = []

for name, genes in SIGNATURES.items():
    present = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]

    present_summary.append({
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

summary = pd.DataFrame(present_summary)
summary.to_csv(OUTDIR / "GSE253355_signature_gene_presence.csv", index=False)

score_cols = [c for c in adata.obs.columns if c.endswith("_score")]

sample_summary = (
    adata.obs[["sample_id"] + score_cols]
    .groupby("sample_id")
    .agg(["mean", "std", "median"])
)

sample_summary.to_csv(OUTDIR / "GSE253355_normal_BM_ecology_scores_by_sample.csv")

cell_scores = adata.obs[["sample_id"] + score_cols].copy()
cell_scores.to_csv(OUTDIR / "GSE253355_normal_BM_ecology_scores_by_cell.csv")

adata.write_h5ad(BASE / "processed" / "GSE253355_normal_BM_reference_ecology_scored.h5ad")

print("Saved:")
print(BASE / "processed" / "GSE253355_normal_BM_reference_ecology_scored.h5ad")
print(OUTDIR / "GSE253355_signature_gene_presence.csv")
print(OUTDIR / "GSE253355_normal_BM_ecology_scores_by_sample.csv")
print(OUTDIR / "GSE253355_normal_BM_ecology_scores_by_cell.csv")
