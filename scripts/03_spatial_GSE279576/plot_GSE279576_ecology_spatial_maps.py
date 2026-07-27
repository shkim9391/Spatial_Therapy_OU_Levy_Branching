from pathlib import Path
import scanpy as sc
import matplotlib.pyplot as plt

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
IN = BASE / "processed" / "ecology_scores"
OUT = BASE / "figures" / "ecology_spatial_maps"
OUT.mkdir(parents=True, exist_ok=True)

PROGRAMS = [
    "AML_blast_like_score",
    "Primitive_like_AML_score",
    "Committed_like_AML_score",
    "Inflammatory_score",
    "CXCL12_CXCR4_axis_score",
    "MSC_stromal_score",
    "Endothelial_vascular_score",
    "Hypoxia_stress_score",
]

for f in sorted(IN.glob("*_visium_ecology_scored.h5ad")):
    sid = f.name.replace("_visium_ecology_scored.h5ad", "")
    print(f"Plotting {sid}")

    adata = sc.read_h5ad(f)

    for program in PROGRAMS:
        if program not in adata.obs.columns:
            print(f"  Missing {program}, skipping")
            continue

        sc.pl.spatial(
            adata,
            color=program,
            img_key="hires",
            spot_size=1.3,
            show=False,
            title=f"{sid}: {program.replace('_score','')}"
        )

        png = OUT / f"{sid}_{program}_spatial_map.png"
        pdf = OUT / f"{sid}_{program}_spatial_map.pdf"

        plt.savefig(png, dpi=600, bbox_inches="tight")
        plt.savefig(pdf, bbox_inches="tight")
        plt.close()

print("Saved maps to:", OUT)
