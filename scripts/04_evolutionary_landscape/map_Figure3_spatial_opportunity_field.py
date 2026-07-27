from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
CTX = BASE / "processed" / "spatial_ecological_contexts"
LAND = BASE / "processed" / "evolutionary_landscape"
FIG = BASE / "figures" / "evolutionary_landscape" / "spatial_opportunity_fields"
FIG.mkdir(parents=True, exist_ok=True)


def get_spatial_image_and_coords(adata, img_key="hires"):
    """
    Return H&E image and scaled spatial coordinates in image-pixel space.
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
    H&E-only panel for paired Fig. 3 spatial layout.
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


def save_opportunity_spots_only_panel(
    adata,
    sid,
    figdir,
    vmin=None,
    vmax=None,
    cmap="magma",
):
    """
    Spot-only evolutionary opportunity field.
    No H&E background, but matched coordinate frame.
    """
    img, coords, _ = get_spatial_image_and_coords(adata, img_key="hires")

    vals = adata.obs["evolutionary_opportunity_z"].astype(float).values

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    sca = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=vals,
        s=10,
        alpha=0.95,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )

    ax.set_title(f"{sid}: spatial evolutionary opportunity", fontsize=13)
    format_spatial_axis(ax, img, coords)

    cbar = plt.colorbar(sca, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Opportunity\n(z-score)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(figdir / f"{sid}_spatial_opportunity_spots_only.png", dpi=600, bbox_inches="tight")
    plt.savefig(figdir / f"{sid}_spatial_opportunity_spots_only.pdf", bbox_inches="tight")
    plt.close()


opp = pd.read_csv(
    LAND / "GSE279576_context_evolutionary_opportunity.csv",
    index_col=0,
)["evolutionary_opportunity_z"].to_dict()

# Use fixed color scale across all Fig. 3 spatial opportunity maps
opp_values = np.array(list(opp.values()), dtype=float)
VMIN = float(np.nanmin(opp_values))
VMAX = float(np.nanmax(opp_values))

for f in sorted(CTX.glob("*_visium_spatial_contexts.h5ad")):
    sid = f.name.replace("_visium_spatial_contexts.h5ad", "")
    print("Mapping opportunity:", sid)

    adata = sc.read_h5ad(f)

    adata.obs["evolutionary_opportunity_z"] = (
        adata.obs["spatial_ecological_context"]
        .astype(str)
        .map(opp)
        .astype(float)
    )

    # Optional: keep old overlay map for reference
    sc.pl.spatial(
        adata,
        color="evolutionary_opportunity_z",
        img_key="hires",
        spot_size=1.5,
        cmap="magma",
        vmin=VMIN,
        vmax=VMAX,
        show=False,
        title=f"{sid}: spatial evolutionary opportunity",
    )

    plt.savefig(
        FIG / f"{sid}_spatial_evolutionary_opportunity_overlay.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.savefig(
        FIG / f"{sid}_spatial_evolutionary_opportunity_overlay.pdf",
        bbox_inches="tight",
    )
    plt.close()

    # New publication-ready separated panels
    save_he_only_panel(adata, sid, FIG)
    save_opportunity_spots_only_panel(
        adata,
        sid,
        FIG,
        vmin=VMIN,
        vmax=VMAX,
        cmap="magma",
    )

    adata.write_h5ad(LAND / f"{sid}_visium_spatial_opportunity.h5ad")

print("Saved opportunity maps to:", FIG)
