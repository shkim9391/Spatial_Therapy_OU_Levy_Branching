from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
OUT = Path("/Spatial_Therapy_OU_Levy_Branching/Figure_4")
OUT.mkdir(exist_ok=True)

INPUT = BASE / "processed" / "spatial_ou_input_table.csv"

NICHE_ORDER = ["S1", "S2", "S3", "S4", "S5"]
NICHE_COLORS = {
    "S1": "#4C78A8",
    "S2": "#59A14F",
    "S3": "#F28E2B",
    "S4": "#E15759",
    "S5": "#B279A2",
}


def standardize_columns(df):
    rename = {}

    if "spatial_x" in df.columns and "x" not in df.columns:
        rename["spatial_x"] = "x"
    if "spatial_y" in df.columns and "y" not in df.columns:
        rename["spatial_y"] = "y"

    if "spatial_ecological_context" in df.columns and "ecological_niche" not in df.columns:
        rename["spatial_ecological_context"] = "ecological_niche"
    if "ecological_context" in df.columns and "ecological_niche" not in df.columns:
        rename["ecological_context"] = "ecological_niche"

    if "tumor_state_score" in df.columns and "latent_state" not in df.columns:
        rename["tumor_state_score"] = "latent_state"

    df = df.rename(columns=rename)

    required = ["sample_id", "x", "y", "ecological_niche", "latent_state"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def add_spatial_pseudo_order(sub):
    coords = sub[["x", "y"]].to_numpy(dtype=float)

    if len(sub) < 3:
        sub = sub.copy()
        sub["pseudo_order"] = sub["x"].rank(method="first")
        return sub

    coords_centered = coords - coords.mean(axis=0, keepdims=True)
    pc1 = PCA(n_components=1).fit_transform(coords_centered).ravel()

    sub = sub.copy()
    sub["pseudo_order"] = pc1
    return sub


def estimate_ou_by_niche(sub):
    x = sub["latent_state"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    mu = np.nanmean(x) if len(x) > 0 else np.nan
    sigma_state = np.nanstd(x) if len(x) > 1 else np.nan

    ordered_sub = add_spatial_pseudo_order(sub)
    ordered = (
        ordered_sub.sort_values("pseudo_order")["latent_state"]
        .to_numpy(dtype=float)
    )
    ordered = ordered[np.isfinite(ordered)]

    theta = np.nan
    rho = np.nan

    if len(ordered) >= 6 and np.nanstd(ordered[:-1]) > 0 and np.nanstd(ordered[1:]) > 0:
        rho = np.corrcoef(ordered[:-1], ordered[1:])[0, 1]

        # OU AR(1)-like approximation requires 0 < rho < 1.
        if np.isfinite(rho) and 0 < rho < 1:
            theta = -np.log(rho)

    return pd.Series({
        "mu": mu,
        "rho": rho,
        "theta": theta,
        "log10_theta": np.log10(theta) if np.isfinite(theta) and theta > 0 else np.nan,
        "sigma_state": sigma_state,
        "n_spots": len(sub),
    })

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

INPUT = BASE / "processed" / "spatial_ou_input_table.csv"

CTX_INPUT = (
    BASE
    / "processed"
    / "spatial_ecological_contexts"
    / "GSE279576_spatial_ecological_context_assignments.csv"
)

LATENT_FEATURES = [
    "AML_blast_like_score",
    "Primitive_like_AML_score",
    "Committed_like_AML_score",
    "HSPC_primitive_score",
    "Myeloid_granulocytic_score",
    "Monocyte_macrophage_score",
    "Inflammatory_score",
    "Hypoxia_stress_score",
]

if not INPUT.exists():
    print("Building spatial OU input table:", INPUT)

    raw = pd.read_csv(CTX_INPUT)

    use_features = [c for c in LATENT_FEATURES if c in raw.columns]
    if len(use_features) < 2:
        raise ValueError(
            "Not enough latent-state features found. "
            f"Available columns include: {raw.columns.tolist()[:30]}"
        )

    X = raw[use_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xz = StandardScaler().fit_transform(X)

    pc1 = PCA(n_components=1, random_state=0).fit_transform(Xz).ravel()

    # Orient latent state so higher values correspond to AML/primitive-like burden.
    anchor_features = [
        c for c in [
            "AML_blast_like_score",
            "Primitive_like_AML_score",
            "Committed_like_AML_score",
            "HSPC_primitive_score",
        ]
        if c in raw.columns
    ]

    if anchor_features:
        anchor = raw[anchor_features].mean(axis=1).to_numpy()
        if np.corrcoef(pc1, anchor)[0, 1] < 0:
            pc1 = -pc1

    ou_input = pd.DataFrame({
        "sample_id": raw["sample_id"],
        "patient_id": raw["sample_id"],
        "spot_id": raw["spot_barcode"],
        "x": raw["spatial_x"],
        "y": raw["spatial_y"],
        "ecological_niche": raw["spatial_ecological_context"],
        "latent_state": pc1,
    })

    ou_input.to_csv(INPUT, index=False)
    print("Saved:", INPUT)

df = pd.read_csv(INPUT)
df = standardize_columns(df)

df = df[df["ecological_niche"].isin(NICHE_ORDER)].copy()
df["ecological_niche"] = pd.Categorical(
    df["ecological_niche"],
    categories=NICHE_ORDER,
    ordered=True,
)

param_df = (
    df.groupby(["sample_id", "ecological_niche"], observed=True)
      .apply(estimate_ou_by_niche, include_groups=False)
      .reset_index()
)

df = df.merge(
    param_df[["sample_id", "ecological_niche", "mu", "theta", "log10_theta"]],
    on=["sample_id", "ecological_niche"],
    how="left",
)

param_df.to_csv(OUT / "Figure4_spatial_ou_parameter_summary.csv", index=False)

# Choose representative sample.
preferred_samples = ["GSM8576301_BM1", "GSM8576303_BM2", "GSM8576306_EM2_v1"]
available = df["sample_id"].dropna().unique().tolist()
example_sample = next((s for s in preferred_samples if s in available), sorted(available)[0])

plot_df = df[df["sample_id"] == example_sample].copy()

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(
    2,
    3,
    width_ratios=[1, 1, 1.1],
    height_ratios=[1, 1],
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
axD = fig.add_subplot(gs[1, 0])
axE = fig.add_subplot(gs[1, 1:])

# A. Spatial mu
scA = axA.scatter(
    plot_df["x"],
    plot_df["y"],
    c=plot_df["mu"],
    s=8,
    cmap="viridis",
    linewidths=0,
)
axA.set_title("Context-level μ projected to spots")
axA.set_aspect("equal")
axA.invert_yaxis()
axA.axis("off")
plt.colorbar(scA, ax=axA, fraction=0.046, pad=0.02, label="Estimated μ")

# B. Spatial OU-like rate index
scB = axB.scatter(
    plot_df["x"],
    plot_df["y"],
    c=plot_df["log10_theta"],
    s=8,
    cmap="magma",
    linewidths=0,
)

axB.set_title("Context-level OU-like rate index projected to spots")
axB.set_aspect("equal")
axB.invert_yaxis()
axB.axis("off")

plt.colorbar(
    scB,
    ax=axB,
    fraction=0.046,
    pad=0.02,
    label=r"$\log_{10}\theta$",
)

# C. Spatial pseudo-trajectories
for niche in NICHE_ORDER:
    sub = plot_df[plot_df["ecological_niche"] == niche].copy()
    if sub.empty:
        continue

    sub = add_spatial_pseudo_order(sub).sort_values("pseudo_order")

    axC.plot(
        np.arange(len(sub)),
        sub["latent_state"],
        color=NICHE_COLORS.get(niche, "gray"),
        alpha=0.75,
        lw=1.0,
        label=niche,
    )

axC.set_title("Spatially ordered latent-state profiles", pad=10)
axC.set_xlabel("Spatial pseudo-order")
axC.set_ylabel("Latent state score")
axC.legend(frameon=False, fontsize=8)

# D. OU attractor basin diagrams
x_min = np.nanpercentile(df["latent_state"], 1)
x_max = np.nanpercentile(df["latent_state"], 99)
xgrid = np.linspace(x_min, x_max, 300)

for niche in NICHE_ORDER:
    sub = param_df[param_df["ecological_niche"] == niche]
    if sub.empty:
        continue

    mu = sub["mu"].mean()
    theta = sub["theta"].mean()

    if not np.isfinite(mu) or not np.isfinite(theta):
        continue

    U = 0.5 * theta * (xgrid - mu) ** 2
    U = U - np.nanmin(U)

    axD.plot(
        xgrid,
        U,
        color=NICHE_COLORS.get(niche, "gray"),
        lw=2,
        label=niche,
    )

axD.set_title("Approximate OU attractor basins")
axD.set_xlabel("Latent tumor/ecological state")
axD.set_ylabel("Relative OU potential")
axD.legend(frameon=False, fontsize=8)

# E. Regional OU parameter comparison
long_param = param_df.melt(
    id_vars=["sample_id", "ecological_niche"],
    value_vars=["mu", "log10_theta"],
    var_name="parameter",
    value_name="value",
)

long_param["parameter"] = long_param["parameter"].replace({
    "mu": "μ",
    "log10_theta": "log10 θ",
})

sns.boxplot(
    data=long_param,
    x="ecological_niche",
    y="value",
    hue="parameter",
    order=NICHE_ORDER,
    ax=axE,
    showfliers=False,
)

sns.stripplot(
    data=long_param,
    x="ecological_niche",
    y="value",
    hue="parameter",
    order=NICHE_ORDER,
    dodge=True,
    size=3,
    alpha=0.5,
    palette="dark:black",
    ax=axE,
)

axE.set_title("Context-level OU-like parameter summaries")
axE.set_xlabel("Spatial ecological context")
axE.set_ylabel("Estimated value")

handles, labels = axE.get_legend_handles_labels()
axE.legend(handles[:2], labels[:2], frameon=False, title="")

for ax, label in zip([axA, axB, axC, axD, axE], list("ABCDE")):
    ax.text(
        -0.08,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )

fig.suptitle(
    "OU-like spatial dynamical summaries across ecological contexts",
    fontsize=16,
    fontweight="bold",
)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.subplots_adjust(wspace=0.35, hspace=0.35)

fig.savefig(OUT / "Figure4_OU_dynamics_across_spatial_contexts.png", dpi=600)
fig.savefig(OUT / "Figure4_OU_dynamics_across_spatial_contexts.pdf")
fig.savefig(OUT / "Figure4_OU_dynamics_across_spatial_contexts.svg")

plt.close(fig)

print("Saved Figure 4 outputs to:", OUT)
print("Representative sample:", example_sample)
