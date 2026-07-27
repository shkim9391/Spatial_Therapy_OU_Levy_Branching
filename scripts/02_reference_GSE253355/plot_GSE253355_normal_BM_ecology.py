from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE253355")
IN = BASE / "processed" / "ecology_scores" / "GSE253355_normal_BM_ecology_scores_by_sample.csv"
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(IN, header=[0, 1], index_col=0)

mean_cols = [c for c in df.columns if c[1] == "mean"]
mat = df[mean_cols].copy()
mat.columns = [c[0].replace("_score", "") for c in mean_cols]

# z-score signatures across samples
zmat = (mat - mat.mean(axis=0)) / mat.std(axis=0)
zmat = zmat.T

fig, ax = plt.subplots(figsize=(16, 9))
im = ax.imshow(zmat.values, aspect="auto")

ax.set_xticks(range(zmat.shape[1]))
ax.set_xticklabels(zmat.columns, rotation=45, ha="right", fontsize=13)

ax.set_yticks(range(zmat.shape[0]))
ax.set_yticklabels(zmat.index, fontsize=13)

#ax.set_title("GSE253355 normal bone marrow ecological reference signatures", fontsize=17)
ax.set_xlabel("Normal bone marrow sample", fontsize=13)
ax.set_ylabel("Ecological program", fontsize=13)

cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Sample-level z-score", fontsize=13)

plt.tight_layout()

png = OUT / "Figure2A_GSE253355_normal_BM_ecology_heatmap.png"
pdf = OUT / "Figure2A_GSE253355_normal_BM_ecology_heatmap.pdf"

plt.savefig(png, dpi=600)
plt.savefig(pdf)
plt.close()

print("Saved:")
print(png)
print(pdf)
