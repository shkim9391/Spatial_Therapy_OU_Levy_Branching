from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("/Spatial_Therapy_OU_Levy_Branching")
OUT = BASE / "Figure_2"
OUT.mkdir(exist_ok=True)

# Image panels except D, which will be replotted directly
paths = {
    "A": BASE / "GSE253355/figures/Figure2A_GSE253355_normal_BM_ecology_heatmap.png",
    "B": BASE / "GSE279576/figures/spatial_ecological_contexts/GSE279576_spatial_ecological_context_feature_heatmap.png",
    "C": BASE / "GSE279576/figures/spatial_ecological_contexts/GSE279576_spatial_ecological_context_PCA.png",
    "E": BASE / "GSE279576/figures/spatial_ecological_contexts/GSM8576301_BM1_spatial_ecological_context_map.png",
    "F": BASE / "GSE279576/figures/spatial_ecological_contexts/GSM8576303_BM2_spatial_ecological_context_map.png",
    "G": BASE / "GSE279576/figures/spatial_ecological_contexts/GSM8576306_EM2_v1_spatial_ecological_context_map.png",
}

imgs = {k: Image.open(v).convert("RGB") for k, v in paths.items()}

fig = plt.figure(figsize=(24, 16))

axes = {
    "A": fig.add_axes([0.05,0.70,0.46,0.28]),
    "B": fig.add_axes([0.52,0.70,0.46,0.28]),
    "C": fig.add_axes([0.02,0.41,0.29,0.27]),
    "D": fig.add_axes([0.38,0.43,0.24,0.24]),
    "E": fig.add_axes([0.63,0.36,0.35,0.32]),
    "F": fig.add_axes([0.03,0.03,0.34,0.30]),
    "G": fig.add_axes([0.39,0.03,0.34,0.30]),
}

for letter in ["A", "B", "C", "E", "F", "G"]:
    ax = axes[letter]
    ax.imshow(imgs[letter])
    ax.axis("off")
    ax.text(-0.03, 1.04, letter, transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="bottom")

# Panel D: replot composition with legend outside
axD = axes["D"]
comp_path = BASE / "GSE279576/processed/spatial_ecological_contexts/GSE279576_spatial_ecological_context_composition_by_site.csv"
site_comp = pd.read_csv(comp_path, index_col=0)

site_comp.plot(
    kind="bar",
    stacked=True,
    ax=axD,
    width=0.62
)

axD.set_title("Spatial ecological context\ncomposition by site", fontsize=10)
axD.set_xlabel("")
axD.set_ylabel("Fraction of spots", fontsize=9)
axD.tick_params(axis="both", labelsize=8)
axD.set_ylim(0, 1.02)

axD.legend(
    title="Context",
    loc="center left",
    bbox_to_anchor=(1.03, 0.5),
    frameon=False,
    fontsize=8,
    title_fontsize=8
)

axD.text(-0.16, 1.08, "D", transform=axD.transAxes,
         fontsize=22, fontweight="bold", va="bottom")

# Panel H
axH = fig.add_axes([0.76, 0.03, 0.20, 0.28])
axH.axis("off")
axH.text(0, 1.0, "H  Proposed biological interpretation",
         fontsize=17, fontweight="bold", va="top")

rows = [
    ("S1", "Blast–committed myeloid /\nhypoxia-stress", "BM-enriched"),
    ("S2", "Primitive AML /\nHSPC-like", "EM-enriched"),
    ("S3", "Immune-suppressed monocyte–\nmacrophage / B-cell", "Shared/BM"),
    ("S4", "Inflammatory stromal–vascular /\nECM niche", "EM-enriched"),
    ("S5", "T/NK immune-active", "EM-enriched"),
]

y = 0.82
for ctx, label, bias in rows:
    axH.text(0.00, y, ctx, fontsize=12, fontweight="bold")
    axH.text(0.14, y, label, fontsize=10)
    axH.text(0.14, y - 0.085, bias, fontsize=9, style="italic")
    y -= 0.17

png = OUT / "Figure2_spatial_ecological_context_discovery_UPDATED_v2.png"
pdf = OUT / "Figure2_spatial_ecological_context_discovery_UPDATED_v2.pdf"
tiff = OUT / "Figure2_spatial_ecological_context_discovery_UPDATED_v2.tiff"

plt.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.15)
plt.savefig(pdf, bbox_inches="tight", pad_inches=0.15)
plt.savefig(tiff, dpi=400, bbox_inches="tight", pad_inches=0.15)
plt.close()

print("Saved:")
print(png)
print(pdf)
print(tiff)
