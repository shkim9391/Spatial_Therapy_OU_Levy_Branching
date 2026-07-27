from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Ellipse, Polygon
from matplotlib.lines import Line2D

# -----------------------------
# Global settings
# -----------------------------
np.random.seed(7)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUTDIR = Path("/Spatial_Therapy_OU_Levy_Branching/Figure_1")
PNG = OUTDIR / "Figure1_spatial_therapy_aware_ou_levy_branching_ecology_UPDATED.png"
PDF = OUTDIR / "Figure1_spatial_therapy_aware_ou_levy_branching_ecology_UPDATED.pdf"
TIFF = OUTDIR / "Figure1_spatial_therapy_aware_ou_levy_branching_ecology_UPDATED.tiff"

# -----------------------------
# Palette
# -----------------------------
COL = {
    "tumor": "#8C1D40",
    "tumor2": "#C44E52",
    "resistant": "#7A0177",
    "immune": "#4C78A8",
    "stromal": "#59A14F",
    "vascular": "#E15759",
    "hypoxic": "#6B6ECF",
    "therapy": "#F28E2B",
    "gray": "#4D4D4D",
    "lightgray": "#F2F2F2",
    "dark": "#222222",
    "basin": "#DDEBF7",
    "background": "#FFFFFF",
}

# -----------------------------
# Helpers
# -----------------------------
def panel_label(ax, label, title):
    ax.text(0.0, 1.04, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    ax.text(0.07, 1.045, title, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")


def arrow(ax, start, end, color="#333333", lw=1.8, rad=0.0, style="->", ms=12, alpha=1.0):
    a = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)
    return a


def cell(ax, xy, r=0.018, color="#8C1D40", ec="white", lw=0.5, alpha=1.0, z=5):
    c = Circle(xy, r, facecolor=color, edgecolor=ec, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(c)
    return c


def draw_niche_band(ax, y, color, label, alpha=0.12):
    ax.add_patch(Rectangle((0.04, y - 0.055), 0.92, 0.11, facecolor=color, alpha=alpha, edgecolor="none"))
    ax.text(0.055, y + 0.032, label, fontsize=8.5, color=COL["dark"], va="center")


def gaussian_landscape(X, Y, wells):
    Z = np.zeros_like(X)
    for x0, y0, amp, sx, sy in wells:
        Z -= amp * np.exp(-(((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2))
    return Z


def draw_landscape(ax, wells, title_labels=None, cmap="viridis", add_contours=True):
    x = np.linspace(-3, 3, 350)
    y = np.linspace(-3, 3, 350)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_landscape(X, Y, wells)
    ax.contourf(X, Y, Z, levels=48, cmap=cmap)
    if add_contours:
        ax.contour(X, Y, Z, levels=12, linewidths=0.35, colors="white", alpha=0.35)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("#555555")
    if title_labels:
        for x0, y0, txt, color in title_labels:
            ax.text(x0, y0, txt, fontsize=8.8, color=color, ha="center", va="center")


# -----------------------------
# Figure layout
# -----------------------------
fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle(
    "Spatial therapy-aware OU–Lévy–branching ecology of pediatric leukemia evolution",
    fontsize=15,
    fontweight="bold",
    y=0.985,
)

axA = fig.add_axes([0.035, 0.56, 0.29, 0.35])
axB = fig.add_axes([0.355, 0.56, 0.29, 0.35])
axC = fig.add_axes([0.675, 0.56, 0.29, 0.35])
axD = fig.add_axes([0.105, 0.10, 0.37, 0.34])
axE = fig.add_axes([0.555, 0.10, 0.37, 0.34])

# ---------------------------------------------------------
# PANEL A: marrow ecosystem + timeline
# ---------------------------------------------------------
axA.set_xlim(0, 1)
axA.set_ylim(0, 1)
axA.axis("off")
panel_label(axA, "A", "Dynamic spatial leukemia evolution")

# timeline top, separated from title
xpos = [0.16, 0.39, 0.62, 0.85]
time_labels = ["Diagnosis", "Therapy", "MRD/remission", "Relapse"]
for i, (x, lab) in enumerate(zip(xpos, time_labels)):
    cell(axA, (x, 0.90), r=0.022, color=[COL["tumor"], COL["therapy"], COL["immune"], COL["resistant"]][i], ec="white", lw=0.8)
    axA.text(x, 0.95, lab, ha="center", va="bottom", fontsize=8.5)
for i in range(len(xpos) - 1):
    arrow(axA, (xpos[i] + 0.03, 0.90), (xpos[i + 1] - 0.03, 0.90), color=COL["gray"], lw=1.5, ms=9)

# marrow niche bands
bands = [
    (0.74, COL["hypoxic"], "Hypoxic niche"),
    (0.58, COL["stromal"], "Stromal-supportive niche"),
    (0.42, COL["immune"], "Immune/inflammatory niche"),
    (0.26, COL["vascular"], "Vascular niche"),
]
for y, c, lab in bands:
    draw_niche_band(axA, y, c, lab)

# vascular sinusoid
xs = np.linspace(0.07, 0.95, 120)
yv = 0.25 + 0.015 * np.sin(xs * 16)
axA.plot(xs, yv, color=COL["vascular"], lw=3.0, alpha=0.65)
axA.plot(xs, yv + 0.025, color=COL["vascular"], lw=0.8, alpha=0.55)
axA.plot(xs, yv - 0.025, color=COL["vascular"], lw=0.8, alpha=0.55)

# niche cells
for _ in range(38):
    x = np.random.uniform(0.10, 0.92)
    y = np.random.choice([0.74, 0.58, 0.42, 0.26]) + np.random.normal(0, 0.028)
    color = np.random.choice([COL["tumor"], COL["tumor2"], COL["immune"], COL["stromal"]], p=[0.45, 0.15, 0.25, 0.15])
    size = np.random.uniform(0.010, 0.018)
    cell(axA, (x, y), r=size, color=color, ec="white", lw=0.25, alpha=0.95)

# resistant emerging pocket near relapse
for loc in [(0.82, 0.55), (0.85, 0.58), (0.88, 0.53), (0.90, 0.60), (0.86, 0.50)]:
    cell(axA, loc, r=0.017, color=COL["resistant"], ec="white", lw=0.3)
axA.add_patch(Ellipse((0.86, 0.55), 0.20, 0.15, facecolor=COL["resistant"], edgecolor="none", alpha=0.10))

# migration/reorganization arrows
for s, e, r in [
    ((0.18, 0.42), (0.29, 0.58), 0.25),
    ((0.34, 0.74), (0.49, 0.42), -0.25),
    ((0.54, 0.26), (0.63, 0.58), 0.28),
    ((0.69, 0.42), (0.82, 0.55), -0.20),
]:
    arrow(axA, s, e, color=COL["gray"], lw=1.4, rad=r, ms=10)

axA.text(0.50, 0.08, "Therapy-driven spatial reconfiguration", ha="center", fontsize=9.5, fontweight="bold")

# ---------------------------------------------------------
# PANEL B: attractor basins
# ---------------------------------------------------------
panel_label(axB, "B", "Spatial ecological attractor basins")
wells_B = [(-1.55, -1.05, 2.2, 1.0, 0.9), (1.45, 1.45, 2.4, 1.0, 1.0), (0.2, -1.55, 1.7, 1.1, 0.8)]
labels_B = [
    (-1.55, -1.05, "Stem-like\nbasin", "white"),
    (1.35, 1.35, "Immune-evasive\nbasin", "white"),
    (0.15, -1.55, "Therapy-persistent\nbasin", "white"),
]
draw_landscape(axB, wells_B, labels_B)
# arrows toward basins (OU pull)
for s, e in [((-2.4, 0.3), (-1.7, -0.85)), ((-0.4, 0.2), (-1.3, -0.95)), ((2.4, 0.2), (1.6, 1.2)), ((-0.2, -2.5), (0.0, -1.75))]:
    arrow(axB, s, e, color="white", lw=1.3, rad=0.15, ms=9, alpha=0.75)
axB.text(-2.75, 2.55, r"local OU pull:  $\Theta(s,t)[\mu(s,t)-X_t(s)]$", color="white", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.25, edgecolor="none"))

# ---------------------------------------------------------
# PANEL C: therapy remodeling before vs after
# ---------------------------------------------------------
axC.set_xlim(0, 1)
axC.set_ylim(0, 1)
axC.axis("off")
panel_label(axC, "C", "Therapy-induced ecological perturbation")

# Two mini landscapes using inset axes inside C
before = axC.inset_axes([0.02, 0.13, 0.40, 0.70])
after = axC.inset_axes([0.58, 0.13, 0.40, 0.70])
for ax, ttl in [(before, "Before therapy"), (after, "After therapy")]:
    ax.set_title(ttl, fontsize=11, pad=3)

wells_before = [(-1.2, -0.7, 2.1, 1.0, 0.9), (1.0, 0.9, 1.7, 1.1, 1.0), (0.6, -1.5, 1.0, 1.0, 0.8)]
wells_after = [(-1.1, -0.6, 0.8, 1.2, 1.0), (1.1, 0.9, 2.6, 1.0, 1.0), (0.35, -1.4, 1.8, 1.2, 0.8)]
draw_landscape(before, wells_before, cmap="YlGnBu", add_contours=False)
draw_landscape(after, wells_after, cmap="YlOrRd", add_contours=False)

# Cells on mini landscapes
for ax, resistant_frac in [(before, 0.10), (after, 0.55)]:
    n = 35
    for _ in range(n):
        if np.random.rand() < resistant_frac:
            cx, cy = np.random.normal(1.0, 0.45), np.random.normal(0.85, 0.45)
            cc = COL["resistant"]
        else:
            cx, cy = np.random.normal(-1.2, 0.55), np.random.normal(-0.7, 0.50)
            cc = COL["tumor"]
        cell(ax, (cx, cy), r=0.07, color=cc, ec="white", lw=0.25, alpha=0.9, z=6)

arrow(axC, (0.44, 0.50), (0.56, 0.50), color=COL["therapy"], lw=2.2, style="simple", ms=18)
axC.text(0.50, 0.58, "therapy", color=COL["therapy"], fontsize=10, ha="center", fontweight="bold")
axC.text(0.50, 0.06, "attractor weakening  →  resistant niche expansion", fontsize=9.5, ha="center", fontweight="bold")

# ---------------------------------------------------------
# PANEL D: landscape + dramatic Levy jump
# ---------------------------------------------------------
panel_label(axD, "D", "Lévy-like spatial escape transitions")
wells_D = [(-1.55, -1.0, 2.2, 1.0, 0.9), (1.45, -1.25, 2.1, 0.9, 0.85), (0.1, 1.25, 0.8, 1.2, 1.0)]
draw_landscape(axD, wells_D, cmap="magma", add_contours=True)
# local OU wandering path then jump
path1 = np.array([[-2.4, -0.2], [-2.0, -0.8], [-1.6, -1.05], [-1.25, -0.85], [-1.55, -1.15]])
axD.plot(path1[:, 0], path1[:, 1], color="white", lw=2.0, alpha=0.9)
arrow(axD, (-1.45, -1.10), (1.10, -1.15), color=COL["therapy"], lw=3.2, rad=-0.22, ms=18)
path2 = np.array([[1.10, -1.15], [1.35, -1.35], [1.65, -1.05]])
axD.plot(path2[:, 0], path2[:, 1], color="white", lw=2.0, alpha=0.9)
axD.text(-1.72, -1.55, "therapy-sensitive\nbasin", color="white", fontsize=8.8, ha="center")
axD.text(1.55, -1.75, "resistant\nbasin", color="white", fontsize=8.8, ha="center")
axD.text(-0.10, -0.65, "large discontinuous\nLévy escape", color="white", fontsize=9.5, ha="center",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.28, edgecolor="none"))

# ---------------------------------------------------------
# PANEL E: spatial branch-mediated relapse recolonization
# ---------------------------------------------------------
axE.set_xlim(0, 1)
axE.set_ylim(0, 1)
axE.axis("off")
panel_label(axE, "E", "Branch-mediated relapse amplification")

# marrow/niche background
for y, c, lab in [(0.75, COL["stromal"], "supportive niche"), (0.52, COL["immune"], "immune-excluded pocket"), (0.29, COL["vascular"], "vascular corridor")]:
    axE.add_patch(Rectangle((0.05, y - 0.08), 0.90, 0.16, facecolor=c, alpha=0.10, edgecolor="none"))
    axE.text(0.06, y + 0.055, lab, fontsize=8.5)

# resistant seed
seed = (0.17, 0.52)
cell(axE, seed, r=0.030, color=COL["resistant"], ec="white", lw=0.6)
axE.text(0.07, 0.61, "resistant escape seed", fontsize=8.8, color=COL["resistant"], fontweight="bold")

# branching waves and cells
levels = [
    [(0.32, 0.62), (0.32, 0.43)],
    [(0.50, 0.73), (0.52, 0.53), (0.50, 0.32)],
    [(0.70, 0.80), (0.72, 0.64), (0.73, 0.47), (0.71, 0.28)],
    [(0.88, 0.73), (0.88, 0.55), (0.88, 0.38)],
]
prev = [seed]
for lev in levels:
    for p in prev:
        targets = sorted(lev, key=lambda q: abs(q[1] - p[1]))[:2]
        for t in targets:
            arrow(axE, p, t, color=COL["resistant"], lw=1.3, rad=np.random.choice([-0.15, 0.15]), ms=8, alpha=0.65)
    for t in lev:
        cell(axE, t, r=0.022, color=COL["resistant"], ec="white", lw=0.45)
    prev = lev

# expanding translucent relapse regions
for xy, w, h, a in [((0.47, 0.54), 0.42, 0.42, 0.07), ((0.70, 0.55), 0.50, 0.55, 0.08), ((0.82, 0.55), 0.32, 0.62, 0.10)]:
    axE.add_patch(Ellipse(xy, w, h, facecolor=COL["resistant"], edgecolor=COL["resistant"], lw=0.8, alpha=a))

# background immune/stromal cells
for _ in range(28):
    x, y = np.random.uniform(0.08, 0.94), np.random.uniform(0.22, 0.82)
    cc = np.random.choice([COL["immune"], COL["stromal"], COL["tumor"]], p=[0.40, 0.35, 0.25])
    cell(axE, (x, y), r=np.random.uniform(0.009, 0.014), color=cc, ec="white", lw=0.2, alpha=0.65, z=3)

axE.text(0.50, 0.08, "spatial recolonization and therapy-resistant ecosystem emergence", ha="center", fontsize=9.5, fontweight="bold")

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COL["tumor"], markeredgecolor='white', markersize=8, label='therapy-sensitive malignant state'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COL["resistant"], markeredgecolor='white', markersize=8, label='resistant/relapse state'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COL["immune"], markeredgecolor='white', markersize=8, label='immune cell'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=COL["stromal"], markeredgecolor='white', markersize=8, label='stromal cell'),
    Line2D([0], [0], color=COL["therapy"], lw=2.5, label='therapy / Lévy escape'),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.015))

# Global flow arrow labels
flow_y = 0.495
for x1, x2, txt in [
    (0.40, 0.35, "local constraints"),
    (0.60, 0.65, "therapy remodeling"),
    (0.45, 0.55, "escape → amplification"),
]:
    fig.patches.append(FancyArrowPatch((x1, flow_y), (x2, flow_y), transform=fig.transFigure,
                                       arrowstyle="->", mutation_scale=12, lw=1.4, color="#777777", alpha=0.75))
    fig.text((x1 + x2) / 2, flow_y + 0.012, txt, ha="center", fontsize=8.5, color="#666666")

# Save
fig.savefig(PNG, dpi=600, bbox_inches="tight")
fig.savefig(PDF, bbox_inches="tight")
fig.savefig(TIFF, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"Saved:\n  {PNG}\n  {PDF}\n  {TIFF}")
