from pathlib import Path
import scanpy as sc
import matplotlib.pyplot as plt

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
IN = BASE / "processed"
OUT = BASE / "figures" / "qc"
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(IN.glob("*_visium_processed.h5ad"))

for f in files:
    sid = f.name.replace("_visium_processed.h5ad", "")
    print("Plotting", sid)

    adata = sc.read_h5ad(f)

    for color in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
        sc.pl.spatial(
            adata,
            color=color,
            img_key="hires",
            spot_size=1.3,
            show=False,
            title=f"{sid}: {color}"
        )

        png = OUT / f"{sid}_{color}_spatial_qc.png"
        pdf = OUT / f"{sid}_{color}_spatial_qc.pdf"

        plt.savefig(png, dpi=600, bbox_inches="tight")
        plt.savefig(pdf, bbox_inches="tight")
        plt.close()

print("Saved spatial QC plots to:", OUT)
