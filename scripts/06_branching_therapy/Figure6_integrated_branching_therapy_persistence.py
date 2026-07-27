from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import FancyArrowPatch


# ============================================================
# Paths
# ============================================================

BASE = Path("/Spatial_Therapy_OU_Levy_Branching")
OUT = BASE / "Figure_6"
OUT.mkdir(exist_ok=True)

BRANCH_INPUT = OUT / "Figure6_branching_input_table.csv"
THERAPY_INPUT = BASE / "Figure_7" / "Figure7_therapy_response_input_table.csv"

FIG_PNG = OUT / "Figure6_integrated_branching_therapy_persistence.png"
FIG_PDF = OUT / "Figure6_integrated_branching_therapy_persistence.pdf"
FIG_SVG = OUT / "Figure6_integrated_branching_therapy_persistence.svg"

BRANCH_SUMMARY_CSV = OUT / "Figure6_branching_amplification_summary.csv"
EDGE_CSV = OUT / "Figure6_branching_network_edges.csv"
THERAPY_SUMMARY_CSV = OUT / "Figure6_therapy_response_summary.csv"


# ============================================================
# Colors
# ============================================================

STATE_COLORS = {
    "resident": "#4C78A8",
    "escaped": "#F28E2B",
    "amplified": "#E15759",
    "other": "#B0B0B0",
}

RESPONSE_ORDER = [
    "relapse-associated",
    "other / unknown",
]

RESPONSE_COLORS = {
    "relapse-associated": "#E15759",
    "other / unknown": "#B0B0B0",
}


# ============================================================
# Helpers
# ============================================================

def infer_response_status(text):
    text = str(text).lower()

    if any(k in text for k in [
        "relapse-associated", "relapse", "relapsed", "resistant",
        "refractory", "rel"
    ]):
        return "relapse-associated"

    return "other / unknown"


def standardize_branching_columns(df):
    rename = {}

    if "state" in df.columns and "cell_state" not in df.columns:
        rename["state"] = "cell_state"
    if "barcode" in df.columns and "cell_id" not in df.columns:
        rename["barcode"] = "cell_id"
    if "tumor_state_score" in df.columns and "latent_state" not in df.columns:
        rename["tumor_state_score"] = "latent_state"
    if "latent_score" in df.columns and "latent_state" not in df.columns:
        rename["latent_score"] = "latent_state"
    if "umap_1" in df.columns and "umap1" not in df.columns:
        rename["umap_1"] = "umap1"
    if "umap_2" in df.columns and "umap2" not in df.columns:
        rename["umap_2"] = "umap2"

    df = df.rename(columns=rename)

    required = [
        "dataset",
        "sample_id",
        "patient_id",
        "cell_id",
        "cell_state",
        "latent_state",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required branching columns: {missing}")

    if "condition_or_timepoint" not in df.columns:
        df["condition_or_timepoint"] = "unspecified"

    df["latent_state"] = pd.to_numeric(df["latent_state"], errors="coerce")
    df = df.dropna(subset=["latent_state"]).copy()

    if "escaped_state" not in df.columns:
        cutoff = df["latent_state"].quantile(0.75)
        df["escaped_state"] = np.where(df["latent_state"] >= cutoff, "escaped", "resident")

    if "opportunity_score" not in df.columns:
        df["opportunity_score"] = 1.0

    if "escape_score" not in df.columns:
        df["escape_score"] = np.where(df["escaped_state"] == "escaped", 1.0, 0.0)

    if "clone_id" not in df.columns:
        df["clone_id"] = df["cell_state"].astype(str)

    if "ecological_context" not in df.columns:
        df["ecological_context"] = df["cell_state"].astype(str)

    df["opportunity_score"] = pd.to_numeric(df["opportunity_score"], errors="coerce").fillna(1.0)
    df["escape_score"] = pd.to_numeric(df["escape_score"], errors="coerce").fillna(0.0)

    return df


def add_embedding_if_missing(df):
    if "umap1" in df.columns and "umap2" in df.columns:
        df["embed1"] = pd.to_numeric(df["umap1"], errors="coerce")
        df["embed2"] = pd.to_numeric(df["umap2"], errors="coerce")
        return df.dropna(subset=["embed1", "embed2"]).copy()

    state_codes = pd.Categorical(df["cell_state"]).codes.astype(float)
    clone_codes = pd.Categorical(df["clone_id"]).codes.astype(float)
    context_codes = pd.Categorical(df["ecological_context"]).codes.astype(float)

    X = pd.DataFrame({
        "latent_state": df["latent_state"].fillna(df["latent_state"].median()),
        "opportunity_score": df["opportunity_score"],
        "escape_score": df["escape_score"],
        "cell_state_code": state_codes,
        "clone_code": clone_codes,
        "context_code": context_codes,
    })

    Xz = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2, random_state=0).fit_transform(Xz)

    df["embed1"] = pcs[:, 0]
    df["embed2"] = pcs[:, 1]

    return df


def compute_branching_summary(df):
    total_by_dataset = df.groupby("dataset").size().rename("dataset_n_cells")
    rows = []

    for (dataset, state), sub in df.groupby(["dataset", "cell_state"]):
        dataset_n = total_by_dataset.loc[dataset]
        abundance = len(sub) / dataset_n

        resident = df[
            (df["dataset"] == dataset)
            & (df["escaped_state"].astype(str) != "escaped")
        ]

        baseline = (
            resident["latent_state"].mean()
            if len(resident) > 0
            else df[df["dataset"] == dataset]["latent_state"].mean()
        )

        latent_mean = sub["latent_state"].mean()
        latent_displacement = abs(latent_mean - baseline)

        escaped_fraction = (sub["escaped_state"].astype(str) == "escaped").mean()
        opportunity_mean = sub["opportunity_score"].mean()
        escape_mean = sub["escape_score"].mean()

        branching_score = (
            abundance
            * (latent_displacement + 1e-6)
            * (1.0 + opportunity_mean)
            * (1.0 + escape_mean)
            * (1.0 + escaped_fraction)
        )

        rows.append({
            "dataset": dataset,
            "cell_state": state,
            "branching_unit": f"{dataset}|{state}",
            "n_cells": len(sub),
            "dataset_n_cells": dataset_n,
            "abundance": abundance,
            "latent_mean": latent_mean,
            "latent_displacement": latent_displacement,
            "escaped_fraction": escaped_fraction,
            "opportunity_mean": opportunity_mean,
            "escape_mean": escape_mean,
            "branching_amplification_score": branching_score,
        })

    return pd.DataFrame(rows).sort_values(
        "branching_amplification_score",
        ascending=False,
    )


def classify_branching_status(df, summary):
    amp_cut = summary["branching_amplification_score"].quantile(0.75)
    amplified_units = set(
        summary.loc[
            summary["branching_amplification_score"] >= amp_cut,
            "branching_unit",
        ]
    )

    df = df.copy()
    df["branching_unit"] = df["dataset"].astype(str) + "|" + df["cell_state"].astype(str)

    df["branching_status"] = "resident"
    df.loc[df["escaped_state"].astype(str) == "escaped", "branching_status"] = "escaped"
    df.loc[df["branching_unit"].isin(amplified_units), "branching_status"] = "amplified"

    return df


def compute_network_edges(df, summary):
    rows = []

    for dataset in df["dataset"].unique():
        s = summary[summary["dataset"] == dataset].copy()

        resident_states = (
            s[s["escaped_fraction"] < 0.5]
            .sort_values("abundance", ascending=False)
            .head(3)
        )

        escaped_states = (
            s[s["escaped_fraction"] >= 0.5]
            .sort_values("branching_amplification_score", ascending=False)
            .head(6)
        )

        if resident_states.empty:
            resident_states = s.head(1)

        if escaped_states.empty:
            escaped_states = s.head(6)

        for _, source in resident_states.iterrows():
            for _, target in escaped_states.iterrows():
                if source["cell_state"] == target["cell_state"]:
                    continue

                weight = (
                    target["branching_amplification_score"]
                    * (1.0 + target["latent_displacement"])
                )

                rows.append({
                    "dataset": dataset,
                    "source": source["cell_state"],
                    "target": target["cell_state"],
                    "weight": weight,
                    "target_abundance": target["abundance"],
                    "target_branching_score": target["branching_amplification_score"],
                })

    return pd.DataFrame(rows)


def standardize_therapy_input(df):
    rename = {}

    if "response" in df.columns and "response_status" not in df.columns:
        rename["response"] = "response_status"
    if "therapy_response" in df.columns and "response_status" not in df.columns:
        rename["therapy_response"] = "response_status"
    if "retention_score" in df.columns and "ou_retention_score" not in df.columns:
        rename["retention_score"] = "ou_retention_score"
    if "escape_score" in df.columns and "levy_escape_score" not in df.columns:
        rename["escape_score"] = "levy_escape_score"
    if "branching_score" in df.columns and "branching_amplification_score" not in df.columns:
        rename["branching_score"] = "branching_amplification_score"

    df = df.rename(columns=rename)

    required = [
        "dataset",
        "patient_id",
        "sample_id",
        "response_status",
        "ou_retention_score",
        "levy_escape_score",
        "branching_amplification_score",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required therapy-response columns: {missing}")

    for c in [
        "ou_retention_score",
        "levy_escape_score",
        "branching_amplification_score",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["response_status"] = df["response_status"].apply(infer_response_status)

    return df


def add_integrated_score(df):
    score_cols = [
        "ou_retention_score",
        "levy_escape_score",
        "branching_amplification_score",
    ]

    zmat = pd.DataFrame(index=df.index)

    for c in score_cols:
        if df[c].std() > 0:
            zmat[c] = zscore(df[c], nan_policy="omit")
        else:
            zmat[c] = 0.0

    df["integrated_olb_risk_score"] = zmat.sum(axis=1)

    return df


# ============================================================
# Plotting functions
# ============================================================

def add_panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )


def plot_branching_framework(ax):
    ax.axis("off")

    xs = [0.18, 0.50, 0.82]
    labels = [
        "Resident\nstate\n(OU-like retention)",
        "Escaped\nstate\n(Lévy-like escape)",
        "Branching\namplification\n(expansion)",
    ]
    colors = [
        STATE_COLORS["resident"],
        STATE_COLORS["escaped"],
        STATE_COLORS["amplified"],
    ]

    for x, label, color in zip(xs, labels, colors):
        circle = plt.Circle(
            (x, 0.58),
            0.13,
            facecolor=color,
            edgecolor="black",
            lw=0.8,
            alpha=0.75,
        )
        ax.add_patch(circle)
        ax.text(x, 0.58, label, ha="center", va="center", fontsize=9)

    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.14, 0.58),
                (x2 - 0.14, 0.58),
                arrowstyle="->",
                mutation_scale=15,
                lw=1.5,
                color="black",
            )
        )

    ax.text(
        0.50,
        0.25,
        "OU-like retention  →  Lévy-like escape  →  branching-like amplification",
        ha="center",
        fontsize=10,
    )

    legend_x = [0.26, 0.50, 0.74]
    legend_labels = ["resident", "escaped", "amplified"]

    for x, lab in zip(legend_x, legend_labels):
        ax.scatter(x - 0.06, 0.10, s=30, color=STATE_COLORS[lab])
        ax.text(x - 0.03, 0.10, lab, va="center", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Branching amplification framework")


def plot_embedding(ax, df):
    for status in ["resident", "escaped", "amplified"]:
        sub = df[df["branching_status"] == status]
        ax.scatter(
            sub["embed1"],
            sub["embed2"],
            s=4,
            alpha=0.65,
            color=STATE_COLORS[status],
            label=status,
            linewidths=0,
        )

    ax.set_title("Escaped and amplified states")
    ax.set_xlabel("UMAP 1" if "umap1" in df.columns else "Embedding 1")
    ax.set_ylabel("UMAP 2" if "umap2" in df.columns else "Embedding 2")
    ax.legend(frameon=False, fontsize=8, markerscale=3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_network(ax, edges, summary):
    ax.set_title("State amplification network")
    ax.axis("off")

    if edges.empty:
        ax.text(0.5, 0.5, "No network edges available", ha="center", va="center")
        return

    G = nx.DiGraph()

    for _, row in edges.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])

    node_scores = (
        summary.groupby("cell_state")["branching_amplification_score"]
        .mean()
        .to_dict()
    )

    top_scores = (
        summary.groupby("cell_state")["branching_amplification_score"]
        .mean()
        .sort_values(ascending=False)
    )

    ranked_nodes = [n for n in top_scores.index if n in G.nodes()]

    pos = {}

    if ranked_nodes:
        pos[ranked_nodes[0]] = (1.0, 0.70)

    middle_nodes = ranked_nodes[1:5]
    for n, y in zip(middle_nodes, np.linspace(0.82, 0.28, max(len(middle_nodes), 1))):
        pos[n] = (0.50, y)

    remaining = [n for n in G.nodes() if n not in pos]
    for n, y in zip(remaining, np.linspace(0.75, 0.20, max(len(remaining), 1))):
        pos[n] = (0.05, y)

    if "other_state" in pos:
        pos["other_state"] = (0.05, 0.18)
    if "not_malignant" in pos:
        pos["not_malignant"] = (0.50, 0.35)

    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    widths = 0.6 + 4.2 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)

    node_values = np.array([node_scores.get(n, 0.0) for n in G.nodes()])
    node_sizes = 350 + 1500 * (
        (node_values - node_values.min())
        / (node_values.max() - node_values.min() + 1e-9)
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=widths,
        alpha=0.45,
        edge_color="gray",
        arrows=True,
        arrowsize=14,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_values,
        cmap="Reds",
        edgecolors="black",
        linewidths=0.8,
    )

    label_pos = {
        n: (x, y + 0.07)
        for n, (x, y) in pos.items()
    }

    nx.draw_networkx_labels(
        G,
        label_pos,
        ax=ax,
        font_size=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.7),
    )

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0.05, 0.95)


def plot_therapy_framework(ax):
    ax.axis("off")

    xs = [0.13, 0.38, 0.63, 0.88]
    labels = [
        "Baseline\necology",
        "Therapy\npressure",
        "Residual\npersistence",
        "Relapse-\nassociated\nexpansion",
    ]
    colors = ["#4C78A8", "#9E9E9E", "#F28E2B", "#E15759"]

    for x, label, color in zip(xs, labels, colors):
        box = plt.Rectangle(
            (x - 0.09, 0.56 - 0.10),
            0.18,
            0.20,
            facecolor=color,
            edgecolor="black",
            lw=0.8,
            alpha=0.75,
        )
        ax.add_patch(box)
        ax.text(x, 0.56, label, ha="center", va="center", fontsize=8)

    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.10, 0.56),
                (x2 - 0.10, 0.56),
                arrowstyle="->",
                mutation_scale=14,
                lw=1.4,
                color="black",
            )
        )

    ax.add_patch(
        FancyArrowPatch(
            (0.05, 0.25),
            (0.95, 0.25),
            arrowstyle="->",
            mutation_scale=12,
            lw=1.2,
            color="black",
        )
    )

    ax.text(
        0.50,
        0.10,
        "OU-like retention + Lévy-like escape + branching-like amplification\n"
        "→ therapy-associated ecological persistence",
        ha="center",
        fontsize=9,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Therapy validation framework")


def plot_metric_by_response(ax, df, metric, title, ylabel, color):
    order = [r for r in RESPONSE_ORDER if r in df["response_status"].unique()]

    sns.boxplot(
        data=df,
        x="response_status",
        y=metric,
        order=order,
        color="#B0B0B0",
        showfliers=False,
        width=0.55,
        ax=ax,
    )

    sns.stripplot(
        data=df,
        x="response_status",
        y=metric,
        order=order,
        color="black",
        size=3,
        alpha=0.55,
        ax=ax,
    )

    ax.set_title(title, color=color, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.spines[["top", "right"]].set_visible(False)


def plot_integrated_heatmap(ax, df):
    show_cols = [
        "HSC-like + progenitor-like fraction",
        "levy_escape_score",
        "branching_amplification_score",
        "integrated_olb_risk_score",
    ]

    plot_df = df.copy()
    plot_df["label"] = plot_df["sample_id"].astype(str)
    plot_df = plot_df.sort_values("integrated_olb_risk_score", ascending=False).head(12)

    X = pd.DataFrame(index=plot_df.index)

    for c in show_cols:
        if plot_df[c].std() > 0:
            X[c] = zscore(plot_df[c], nan_policy="omit")
        else:
            X[c] = 0.0

    sns.heatmap(
        X,
        cmap="vlag",
        center=0,
        yticklabels=plot_df["label"],
        xticklabels=[
            "Primitive-state\nretention",
            "Lévy\nescape",
            "Branching\namplification",
            "Integrated\ncomposite",
        ],
        ax=ax,
        cbar_kws={"label": "z-score"},
    )

    ax.set_title("Integrated ecological composite profiles")
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_final_conceptual_model(ax):
    ax.axis("off")

    x = np.linspace(0, 1, 500)

    centers = [0.14, 0.38, 0.62, 0.86]
    colors = ["#4C78A8", "#F28E2B", "#E15759", "#7A3E9D"]
    titles = [
        "O-like retention\n(attractor)",
        "Lévy escape\n(rare jump)",
        "Branching amplification\n(expansion)",
        "Therapy persistence\n(relapse niche)",
    ]

    for c, col in zip(centers, colors):
        for level in np.linspace(0.06, 0.30, 8):
            y = 0.18 + 2.6 * (x - c) ** 2 + level
            mask = np.abs(x - c) < 0.19
            ax.plot(x[mask], y[mask], color=col, lw=0.7, alpha=0.65)

    dot_sets = {
        0: [(0.14, 0.26, 220)],
        1: [(0.35, 0.55, 80), (0.38, 0.65, 130), (0.41, 0.52, 70)],
        2: [(0.58, 0.35, 80), (0.61, 0.42, 120), (0.63, 0.50, 160),
            (0.66, 0.36, 90), (0.60, 0.27, 70), (0.65, 0.28, 90)],
        3: [(0.83, 0.34, 90), (0.86, 0.43, 130), (0.88, 0.33, 100),
            (0.90, 0.52, 160), (0.84, 0.24, 80)],
    }

    for i, pts in dot_sets.items():
        for px, py, ps in pts:
            ax.scatter(px, py, s=ps, color=colors[i], edgecolor="white", lw=0.8, zorder=3)

    arrows = [
        ((0.20, 0.32), (0.34, 0.55)),
        ((0.43, 0.58), (0.56, 0.43)),
        ((0.68, 0.40), (0.80, 0.39)),
    ]

    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=16,
                lw=1.5,
                color="black",
            )
        )

    for c, title, col in zip(centers, titles, colors):
        ax.text(c, 0.95, title, ha="center", va="top", fontsize=10, color=col)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.0)
    ax.set_title("Final conceptual model")


# ============================================================
# Main
# ============================================================

def main():
    if not BRANCH_INPUT.exists():
        raise FileNotFoundError(
            f"Missing branching input table:\n{BRANCH_INPUT}\n\n"
            "Please generate Figure6_branching_input_table.csv first."
        )

    if not THERAPY_INPUT.exists():
        raise FileNotFoundError(
            f"Missing therapy-response input table:\n{THERAPY_INPUT}\n\n"
            "Please generate Figure7_therapy_response_input_table.csv first."
        )

    # Branching data
    branch_df = pd.read_csv(BRANCH_INPUT)
    branch_df = standardize_branching_columns(branch_df)
    branch_df = add_embedding_if_missing(branch_df)

    branch_summary = compute_branching_summary(branch_df)
    branch_df = classify_branching_status(branch_df, branch_summary)
    edges = compute_network_edges(branch_df, branch_summary)

    branch_summary.to_csv(BRANCH_SUMMARY_CSV, index=False)
    edges.to_csv(EDGE_CSV, index=False)

    # Therapy-response data
    therapy_df = pd.read_csv(THERAPY_INPUT)
    therapy_df = standardize_therapy_input(therapy_df)
    therapy_df = add_integrated_score(therapy_df)
    therapy_df.to_csv(THERAPY_SUMMARY_CSV, index=False)

    # Figure layout
    sns.set_theme(style="white", font_scale=0.9)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.05, 1.05, 1.15],
        width_ratios=[1.10, 1.20, 1.30],
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    axD = fig.add_subplot(gs[1, 0])

    subE = gs[1, 1:3].subgridspec(1, 3, wspace=0.40)
    axE1 = fig.add_subplot(subE[0, 0])
    axE2 = fig.add_subplot(subE[0, 1])
    axE3 = fig.add_subplot(subE[0, 2])

    axF = fig.add_subplot(gs[2, 0])
    axG = fig.add_subplot(gs[2, 1:3])

    plot_branching_framework(axA)
    plot_embedding(axB, branch_df)
    plot_network(axC, edges, branch_summary)

    plot_therapy_framework(axD)

    plot_metric_by_response(
        axE1,
        therapy_df,
        "ou_retention_score",
        "OU retention",
        "OU retention score",
        "#4C78A8",
    )

    plot_metric_by_response(
        axE2,
        therapy_df,
        "levy_escape_score",
        "Lévy-like escape",
        "Lévy escape score",
        "#F28E2B",
    )

    plot_metric_by_response(
        axE3,
        therapy_df,
        "branching_amplification_score",
        "Branching amplification",
        "Branching amplification score",
        "#E15759",
    )

    axE1.text(
        -0.35,
        1.15,
        "E",
        transform=axE1.transAxes,
        fontsize=20,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )

    axE1.text(
        0.0,
        1.15,
        "Ecological summary metrics by therapy response",
        transform=axE1.transAxes,
        fontsize=13,
        ha="left",
        va="bottom",
    )

    plot_integrated_heatmap(axF, therapy_df)
    plot_final_conceptual_model(axG)

    for ax, label in zip([axA, axB, axC, axD, axF, axG], ["A", "B", "C", "D", "F", "G"]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Integrated branching amplification and therapy-associated ecological persistence",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.subplots_adjust(wspace=0.35, hspace=0.55)

    fig.savefig(FIG_PNG, dpi=600)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)

    print("Saved:")
    print(FIG_PNG)
    print(FIG_PDF)
    print(FIG_SVG)
    print(BRANCH_SUMMARY_CSV)
    print(EDGE_CSV)
    print(THERAPY_SUMMARY_CSV)


if __name__ == "__main__":
    main()
