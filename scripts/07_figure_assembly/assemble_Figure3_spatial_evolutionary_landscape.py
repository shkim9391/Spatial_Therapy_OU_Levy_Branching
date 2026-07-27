from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx

def crop_whitespace(img, bg=(255, 255, 255), tol=245):
    arr = np.array(img)
    mask = (arr[:, :, 0] < tol) | (arr[:, :, 1] < tol) | (arr[:, :, 2] < tol)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img.crop((x0, y0, x1, y1))

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
OUT = Path("/Spatial_Therapy_OU_Levy_Branching/Figure_3")
OUT.mkdir(exist_ok=True)

LAND = BASE / "processed" / "evolutionary_landscape"
FIG = BASE / "figures" / "evolutionary_landscape"
OPP = FIG / "spatial_opportunity_fields"

opp = pd.read_csv(LAND / "GSE279576_context_evolutionary_opportunity.csv", index_col=0)
trans = pd.read_csv(LAND / "GSE279576_context_transition_proxy.csv", index_col=0)

context_labels = {
    "S1": "Blast-committed myeloid /\nhypoxia-stress",
    "S2": "Primitive AML /\nHSPC-like",
    "S3": "Immune-suppressed\nmonocyte-macrophage /\nB-cell",
    "S4": "Inflammatory stromal-vascular /\nECM niche",
    "S5": "T/NK\nimmune-active",
}

context_colors = {
    "S1": "#F1C40F",
    "S2": "#1F9E89",
    "S3": "#2AA876",
    "S4": "#8CC63E",
    "S5": "#6A1B9A",
}

fig = plt.figure(figsize=(24, 16))

# -------------------------
# Panel A: network
# -------------------------
axA = fig.add_axes([0.03, 0.57, 0.30, 0.38])
G = nx.DiGraph()
contexts = list(opp.index)

for ctx in contexts:
    G.add_node(ctx)

for a in contexts:
    row = trans.loc[a].drop(a)
    top = row.sort_values(ascending=False).head(2)
    for b, w in top.items():
        if w > 0.05:
            G.add_edge(a, b, weight=float(w))

pos = {
    "S1": (-1.1, 0.2),
    "S2": (0.0, 2.0),
    "S3": (1.1, 0.2),
    "S4": (0.1, -1.0),
    "S5": (-1.3, -0.9),
}

weights = [G[u][v]["weight"] for u, v in G.edges()]
nx.draw_networkx_edges(
    G, pos, ax=axA,
    width=[8*w for w in weights],
    alpha=0.45,
    arrows=True,
    arrowsize=22,
    edge_color="gray",
    connectionstyle="arc3,rad=0.08"
)

for ctx in contexts:
    nx.draw_networkx_nodes(
        G, pos, nodelist=[ctx],
        node_size=3300,
        node_color=context_colors[ctx],
        edgecolors="black",
        linewidths=1.5,
        ax=axA
    )

nx.draw_networkx_labels(G, pos, ax=axA, font_size=27, font_weight="bold")

label_offsets = {
    "S1": (-0.30, 0.40),
    "S2": (0.45, 0.05),
    "S3": (0.15, 0.47),
    "S4": (0.70, -0.10),
    "S5": (-0.27, 0.40),
}

for ctx, (x, y) in pos.items():
    dx, dy = label_offsets[ctx]
    axA.text(
        x + dx, y + dy,
        context_labels[ctx],
        ha="center",
        va="center",
        fontsize=10
    )

axA.set_title("A  Context-level evolutionary landscape", loc="left", fontsize=13, fontweight="bold")
axA.axis("off")

# -------------------------
# Panel B: opportunity bar
# -------------------------
axB = fig.add_axes([0.4, 0.64, 0.27, 0.25])
opp_sorted = opp.sort_values("evolutionary_opportunity_z", ascending=False)
bars = axB.bar(
    opp_sorted.index,
    opp_sorted["evolutionary_opportunity_z"],
    color=[context_colors[c] for c in opp_sorted.index],
    edgecolor="black",
    linewidth=0.7
)
axB.axhline(0, color="black", linewidth=0.8)
axB.set_ylabel("Evolutionary opportunity z-score", fontsize=10)
axB.set_xlabel("Spatial ecological context", fontsize=10)
axB.set_title("B  Evolutionary opportunity by spatial context", loc="left", fontsize=15, fontweight="bold")
axB.tick_params(labelsize=10)
for bar in bars:
    h = bar.get_height()
    axB.text(bar.get_x() + bar.get_width()/2, h + (0.05 if h >= 0 else -0.12), f"{h:.2f}",
             ha="center", va="bottom" if h >= 0 else "top", fontsize=9)

# -------------------------
# Panels C-E: opportunity maps
# -------------------------
map_files = {
    "C  BM1: spatial evolutionary opportunity field": OPP / "GSM8576301_BM1_spatial_evolutionary_opportunity.png",
    "D  BM2: spatial evolutionary opportunity field": OPP / "GSM8576303_BM2_spatial_evolutionary_opportunity.png",
    "E  EM2_v1: spatial evolutionary opportunity field": OPP / "GSM8576306_EM2_v1_spatial_evolutionary_opportunity.png",
}

positions = [
    [0.77, 0.58, 0.21, 0.35],
    [0.02, 0.08, 0.28, 0.34],
    [0.31, 0.08, 0.27, 0.34],
]

for (title, path), rect in zip(map_files.items(), positions):
    ax = fig.add_axes(rect)
    img = crop_whitespace(Image.open(path).convert("RGB"))
    ax.imshow(img, aspect="auto")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.text(
        0.02, -0.08,
        "Higher opportunity (yellow/orange) → permissive niches\nLower opportunity (purple/black) → constrained niches",
        transform=ax.transAxes,
        fontsize=12,
        color="darkred"
    )

# -------------------------
# Panel F: conceptual OU-Lévy-branching schematic
# -------------------------
axF = fig.add_axes([0.60, -0.03, 0.38, 0.43])
axF.axis("off")
axF.set_xlim(0, 1)
axF.set_ylim(0, 1)
axF.set_title("F  Conceptual model: OU-Lévy-branching dynamics in ecological space",
              loc="left", fontsize=14, fontweight="bold")

xs = [0.10, 0.365, 0.635, 0.90]
titles = [
    "1) Local stabilization\n(OU attraction)",
    "2) Stochastic escape\n(Lévy jump)",
    "3) Branch amplification\nand expansion",
    "4) Recolonization of\nnew ecological context",
]
colors = ["#1f77b4", "#6A1B9A", "#d62728", "#2ca02c"]

for i, x in enumerate(xs):
    circle = plt.Circle((x, 0.58), 0.10, fill=False, linestyle="--", linewidth=1.2, color=colors[i])
    axF.add_patch(circle)
    axF.text(x, 0.82, titles[i], ha="center", va="top", fontsize=11, color=colors[i], fontweight="bold")

    # toy landscape contours
    for r in [0.02, 0.04, 0.06, 0.08]:
        e = plt.Circle((x, 0.55), r, fill=False, linewidth=0.5, alpha=0.5, color=colors[i])
        axF.add_patch(e)

    # dots
    rng = np.random.default_rng(i)
    for _ in range(7 if i >= 2 else 3):
        px = x + rng.normal(0, 0.025)
        py = 0.55 + rng.normal(0, 0.025)
        axF.plot(px, py, "o", color=colors[i], markersize=4)

for i in range(3):
    axF.annotate(
        "",
        xy=(xs[i+1] - 0.12, 0.58),
        xytext=(xs[i] + 0.12, 0.58),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=2.2,
            color="black",
            shrinkA=0,
            shrinkB=0,
            mutation_scale=18
        ),
        zorder=20
    )

axF.text(0.5, 0.28,
         "Solid arrows = likely local ecological transitions\nDashed contours = ecological opportunity basins\nDots = cell/clone populations or subclonal lineages",
         ha="center", fontsize=12,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8))

# -------------------------
# Main title
# -------------------------
#fig.suptitle(
#    "Figure 3. Spatial ecological contexts form an evolutionary landscape for AML persistence, escape, and expansion",
#    fontsize=18,
#    fontweight="bold",
#    y=0.975
#)

png = OUT / "Figure3_spatial_evolutionary_landscape_DRAFT.png"
pdf = OUT / "Figure3_spatial_evolutionary_landscape_DRAFT.pdf"
tiff = OUT / "Figure3_spatial_evolutionary_landscape_DRAFT.tiff"

plt.savefig(png, dpi=300)
plt.savefig(pdf)
plt.savefig(tiff, dpi=300)
plt.close()

print("Saved:")
print(png)
print(pdf)
print(tiff)
