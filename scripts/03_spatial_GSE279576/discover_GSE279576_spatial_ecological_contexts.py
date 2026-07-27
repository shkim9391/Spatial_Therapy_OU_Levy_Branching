from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
IN = BASE / "processed" / "ecology_scores"
OUT = BASE / "processed" / "spatial_ecological_contexts"
FIG = BASE / "figures" / "spatial_ecological_contexts"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "AML_blast_like_score",
    "Primitive_like_AML_score",
    "Committed_like_AML_score",
    "HSPC_primitive_score",
    "Myeloid_granulocytic_score",
    "Monocyte_macrophage_score",
    "Inflammatory_score",
    "CXCL12_CXCR4_axis_score",
    "MSC_stromal_score",
    "Endothelial_vascular_score",
    "Osteolineage_endosteal_score",
    "Adipocytic_score",
    "ECM_matrix_score",
    "Hypoxia_stress_score",
    "T_NK_score",
    "B_cell_score",
    "Immune_suppression_score",
]

CONTEXT_COLORS = {
    "S1": "#1f77b4",
    "S2": "#ff7f0e",
    "S3": "#2ca02c",
    "S4": "#d62728",
    "S5": "#9467bd",
}


def get_spatial_image_and_coords(adata, img_key="hires"):
    """
    Return H&E image and scaled spatial coordinates in image-pixel space.
    This gives matched coordinate limits for H&E-only and spot-only panels.
    """
    if "spatial" not in adata.uns or len(adata.uns["spatial"]) == 0:
        return None, adata.obsm["spatial"].copy(), None

    library_id = list(adata.uns["spatial"].keys())[0]
    spatial_info = adata.uns["spatial"][library_id]

    img = None
    scale = 1.0

    if "images" in spatial_info and img_key in spatial_info["images"]:
        img = spatial_info["images"][img_key]

    if "scalefactors" in spatial_info:
        if img_key == "hires":
            scale = spatial_info["scalefactors"].get("tissue_hires_scalef", 1.0)
        elif img_key == "lowres":
            scale = spatial_info["scalefactors"].get("tissue_lowres_scalef", 1.0)

    coords = adata.obsm["spatial"].copy() * scale

    return img, coords, library_id


def format_spatial_axis(ax, img, coords):
    """
    Use the same spatial frame for H&E and spot-only panels.
    """
    ax.set_xlabel("spatial1", fontsize=12)
    ax.set_ylabel("spatial2", fontsize=12)
    ax.tick_params(labelsize=9)

    if img is not None:
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
    else:
        ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
        ax.set_ylim(coords[:, 1].max(), coords[:, 1].min())

    ax.set_aspect("equal")


def save_he_only_panel(adata, sid, figdir):
    """
    H&E-only panel for publication layout.
    """
    img, coords, _ = get_spatial_image_and_coords(adata, img_key="hires")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    if img is not None:
        ax.imshow(img)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.1)

    ax.set_title(f"{sid}: H&E morphology", fontsize=13)
    format_spatial_axis(ax, img, coords)

    plt.tight_layout()
    plt.savefig(figdir / f"{sid}_HE_only.png", dpi=600, bbox_inches="tight")
    plt.savefig(figdir / f"{sid}_HE_only.pdf", bbox_inches="tight")
    plt.close()


def save_context_spots_only_panel(adata, sid, figdir):
    """
    Spot-only spatial ecological context map.
    No H&E background, but matched coordinate frame.
    """
    img, coords, _ = get_spatial_image_and_coords(adata, img_key="hires")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    obs = adata.obs["spatial_ecological_context"].astype(str)

    for ctx in sorted(obs.unique()):
        idx = obs.values == ctx
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=10,
            alpha=0.9,
            c=CONTEXT_COLORS.get(ctx, "gray"),
            label=ctx,
            edgecolors="none",
        )

    ax.set_title(f"{sid}: spatial ecological contexts", fontsize=13)
    format_spatial_axis(ax, img, coords)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        markerscale=2.0,
        fontsize=10,
        title=None,
    )

    plt.tight_layout()
    plt.savefig(figdir / f"{sid}_spatial_context_spots_only.png", dpi=600, bbox_inches="tight")
    plt.savefig(figdir / f"{sid}_spatial_context_spots_only.pdf", bbox_inches="tight")
    plt.close()


rows = []
adatas = {}

for f in sorted(IN.glob("*_visium_ecology_scored.h5ad")):
    sid = f.name.replace("_visium_ecology_scored.h5ad", "")
    print("Loading", sid)

    adata = sc.read_h5ad(f)
    adatas[sid] = adata

    use_features = [c for c in FEATURES if c in adata.obs.columns]
    df = adata.obs[["sample_id", "site", "version", "array_row", "array_col"] + use_features].copy()
    df["spot_barcode"] = df.index.astype(str)
    df["spatial_x"] = adata.obsm["spatial"][:, 0]
    df["spatial_y"] = adata.obsm["spatial"][:, 1]
    rows.append(df)

meta = pd.concat(rows, axis=0, ignore_index=True)
features = [c for c in FEATURES if c in meta.columns]

X = meta[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
Xz = StandardScaler().fit_transform(X)

pca = PCA(n_components=min(8, Xz.shape[1]), random_state=0)
PC = pca.fit_transform(Xz)

for i in range(PC.shape[1]):
    meta[f"EcoPC{i+1}"] = PC[:, i]

# Main context solution
K = 5
km = KMeans(n_clusters=K, random_state=0, n_init=50)
labels = km.fit_predict(PC[:, :5])

# Order contexts by AML/primitive/stromal/inflammatory pattern for interpretability
meta["context_raw"] = labels
context_means_tmp = meta.groupby("context_raw")[features].mean()

order_score = (
    context_means_tmp.get("AML_blast_like_score", 0)
    + context_means_tmp.get("Primitive_like_AML_score", 0)
    + 0.5 * context_means_tmp.get("Inflammatory_score", 0)
    + 0.25 * context_means_tmp.get("CXCL12_CXCR4_axis_score", 0)
)

ordered = list(order_score.sort_values(ascending=False).index)
rename = {old: f"S{i+1}" for i, old in enumerate(ordered)}
meta["spatial_ecological_context"] = meta["context_raw"].map(rename)

# Save matrices
meta.to_csv(OUT / "GSE279576_spatial_ecological_context_assignments.csv", index=False)

context_means = meta.groupby("spatial_ecological_context")[features].mean()
context_means_z = (context_means - context_means.mean(axis=0)) / context_means.std(axis=0)
context_means.to_csv(OUT / "GSE279576_spatial_ecological_context_feature_means.csv")
context_means_z.to_csv(OUT / "GSE279576_spatial_ecological_context_feature_means_z.csv")

composition = pd.crosstab(meta["sample_id"], meta["spatial_ecological_context"], normalize="index")
composition.to_csv(OUT / "GSE279576_spatial_ecological_context_composition_by_sample.csv")

site_composition = pd.crosstab(meta["site"], meta["spatial_ecological_context"], normalize="index")
site_composition.to_csv(OUT / "GSE279576_spatial_ecological_context_composition_by_site.csv")

pd.DataFrame({
    "PC": [f"EcoPC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "explained_variance_ratio": pca.explained_variance_ratio_,
}).to_csv(OUT / "GSE279576_ecological_PCA_variance.csv", index=False)

# Heatmap of context-defining features
fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(context_means_z.values, aspect="auto")

ax.set_xticks(range(len(features)))
ax.set_xticklabels(
    [x.replace("_score", "") for x in features],
    rotation=60,
    ha="right",
    fontsize=10,
)

ax.set_yticks(range(context_means_z.shape[0]))
ax.set_yticklabels(context_means_z.index, fontsize=10)

cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Context-level z-score")

plt.tight_layout()
plt.savefig(FIG / "GSE279576_spatial_ecological_context_feature_heatmap.png", dpi=600)
plt.savefig(FIG / "GSE279576_spatial_ecological_context_feature_heatmap.pdf")
plt.close()

# PCA scatter
fig, ax = plt.subplots(figsize=(8.2, 5.8))

for ctx, sub in meta.groupby("spatial_ecological_context"):
    ax.scatter(
        sub["EcoPC1"],
        sub["EcoPC2"],
        s=5,
        alpha=0.6,
        label=ctx,
        c=CONTEXT_COLORS.get(ctx, "gray"),
    )

ax.set_xlabel("EcoPC1", fontsize=13)
ax.set_ylabel("EcoPC2", fontsize=13)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    markerscale=3,
    fontsize=11,
    frameon=False,
    title=None,
)

plt.tight_layout()
plt.savefig(FIG / "GSE279576_spatial_ecological_context_PCA.png", dpi=600, bbox_inches="tight")
plt.savefig(FIG / "GSE279576_spatial_ecological_context_PCA.pdf", bbox_inches="tight")
plt.close()

# Composition by site
fig, ax = plt.subplots(figsize=(5.8, 4.2))

site_composition.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    width=0.55,
    color=[CONTEXT_COLORS.get(c, "gray") for c in site_composition.columns],
)

ax.set_ylabel("Fraction of spots", fontsize=13)
ax.set_xlabel("Site", fontsize=13)
ax.tick_params(axis="both", labelsize=12)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=11,
    title=None,
)

plt.tight_layout()
plt.savefig(FIG / "GSE279576_context_composition_by_site.png", dpi=600, bbox_inches="tight")
plt.savefig(FIG / "GSE279576_context_composition_by_site.pdf", bbox_inches="tight")
plt.close()

# Map contexts back onto tissue and generate paired H&E-only / spot-only panels
for sid, adata in adatas.items():
    sub = meta[meta["sample_id"] == sid].copy()
    mapper = dict(zip(sub["spot_barcode"], sub["spatial_ecological_context"]))

    adata.obs["spatial_ecological_context"] = (
        adata.obs.index.astype(str).map(mapper).astype("category")
    )

    # Optional: keep old overlay map for reference
    sc.pl.spatial(
        adata,
        color="spatial_ecological_context",
        img_key="hires",
        spot_size=1.3,
        show=False,
        title=f"{sid}: spatial ecological contexts",
        palette=CONTEXT_COLORS,
    )
    plt.savefig(FIG / f"{sid}_spatial_ecological_context_overlay.png", dpi=600, bbox_inches="tight")
    plt.savefig(FIG / f"{sid}_spatial_ecological_context_overlay.pdf", bbox_inches="tight")
    plt.close()

    # New publication-ready separated panels
    save_he_only_panel(adata, sid, FIG)
    save_context_spots_only_panel(adata, sid, FIG)

    adata.write_h5ad(OUT / f"{sid}_visium_spatial_contexts.h5ad")

print("Saved outputs to:")
print(OUT)
print(FIG)
