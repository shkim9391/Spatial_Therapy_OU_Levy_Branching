from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import FancyArrowPatch


BASE = Path("/Spatial_Therapy_OU_Levy_Branching")
OUT = BASE / "Figure_6"
OUT.mkdir(exist_ok=True)

INPUT = OUT / "Figure6_branching_input_table.csv"

FIG_PNG = OUT / "Figure6_branching_amplification.png"
FIG_PDF = OUT / "Figure6_branching_amplification.pdf"
FIG_SVG = OUT / "Figure6_branching_amplification.svg"

SUMMARY_CSV = OUT / "Figure6_branching_amplification_summary.csv"
EDGE_CSV = OUT / "Figure6_branching_network_edges.csv"

STATE_COLORS = {
    "resident": "#4C78A8",
    "escaped": "#F28E2B",
    "amplified": "#E15759",
    "other": "#B0B0B0",
}

def build_input_table_from_h5ad_if_missing():
    import scanpy as sc

    if INPUT.exists():
        return

    print("Building Figure 6 input table:", INPUT)

    dataset_dirs = [
        Path("/Ecotype_OU_Branching/GSE235923"),
    ]

    rows = []

    state_candidates = [
        "pred_malignant_coarse",
        "pred_celltype",
        "pred_broad",
        "pred_malignant",
        "predicted_malignant_coarse",
        "predicted_celltype",
        "predicted_broad",
        "predicted_malignant",
        "predicted_cell_type",
        "predicted.celltype",
        "malignant_coarse",
        "predicted_core4",
        "core4",
        "label_transfer",
        "transferred_label",
        "reference_label",
        "cell_state",
        "cell_type",
        "celltype",
        "annotation",
        "seurat_clusters",
        "leiden",
        "cluster",
    ]

    patient_candidates = [
        "patient_id",
        "patient",
        "donor",
        "case",
        "subject",
        "orig.ident",
    ]

    condition_candidates = [
        "condition_or_timepoint",
        "condition",
        "timepoint",
        "phase",
        "status",
        "disease_status",
        "sample_type",
    ]

    latent_candidates = [
        "latent_state",
        "tumor_state_score",
        "leukemia_score",
        "AML_blast_like_score",
        "Primitive_like_AML_score",
        "stemness_score",
        "malignant_score",
    ]

    for ddir in dataset_dirs:
        if not ddir.exists():
            print("Missing dataset directory:", ddir)
            continue

        dataset = ddir.name

        preferred_files = [
            ddir / "derived_secondary_calibration" / "gse235923_dx_secondary_calibration_labeled_by_gse235063.h5ad",
        ]
        
        h5ad_files = [f for f in preferred_files if f.exists()]
        
        if not h5ad_files:
            h5ad_files = sorted(ddir.rglob("*labeled*.h5ad"))
            
        if not h5ad_files:
            print("No h5ad files found in:", ddir)
            continue

        for f in h5ad_files:
            print("Loading:", f)
            adata = sc.read_h5ad(f)

            obs = adata.obs.copy()
            print("\n=== OBS COLUMNS FOR", f.name, "===")
            for c in obs.columns:
                if any(k in c.lower() for k in [
                    "pred", "label", "cell", "type", "broad", "malignant",
                    "coarse", "cluster", "leiden", "annotation"
                ]):
                    print(c)
            print("====================================\n")
            obs["cell_id"] = obs.index.astype(str)
            obs["dataset"] = dataset
            obs["sample_id"] = obs.get("sample_id", f.stem).astype(str) if "sample_id" in obs.columns else f.stem

            patient_col = next((c for c in patient_candidates if c in obs.columns), None)
            if patient_col is not None:
                obs["patient_id"] = obs[patient_col].astype(str)
            else:
                obs["patient_id"] = obs["sample_id"].astype(str)

            state_col = next((c for c in state_candidates if c in obs.columns), None)
            if state_col is not None:
                obs["cell_state"] = obs[state_col].astype(str)
            else:
                obs["cell_state"] = obs["sample_id"].astype(str)
                print("No biological state label found; using sample_id as temporary cell_state")

            condition_col = next((c for c in condition_candidates if c in obs.columns), None)
            if condition_col is not None:
                obs["condition_or_timepoint"] = obs[condition_col].astype(str)
            else:
                obs["condition_or_timepoint"] = "unspecified"

            latent_col = next((c for c in latent_candidates if c in obs.columns), None)
            if latent_col is not None:
                obs["latent_state"] = pd.to_numeric(obs[latent_col], errors="coerce")
            else:
                # Safer fallback: avoid reprocessing adata.X, which may already be log-transformed.
                # Prefer existing PCA/UMAP coordinates if present; otherwise use state labels as
                # a phenotypic ordering proxy for draft visualization.
                if "X_pca" in adata.obsm:
                    obs["latent_state"] = adata.obsm["X_pca"][:, 0]
                    print("Using adata.obsm['X_pca'][:, 0] as latent_state")
            
                elif "X_umap" in adata.obsm:
                    obs["latent_state"] = adata.obsm["X_umap"][:, 0]
                    print("Using adata.obsm['X_umap'][:, 0] as latent_state")
            
                else:
                    numeric_cols = [
                        c for c in obs.columns
                        if pd.api.types.is_numeric_dtype(obs[c])
                        and any(k in c.lower() for k in [
                            "score", "pc", "prob", "prediction", "malignant", "blast", "primitive"
                        ])
                    ]
            
                    if numeric_cols:
                        use_col = numeric_cols[0]
                        obs["latent_state"] = pd.to_numeric(obs[use_col], errors="coerce")
                        print(f"Using numeric obs column '{use_col}' as latent_state")
                    else:
                        codes = pd.Categorical(obs["cell_state"]).codes.astype(float)
                        obs["latent_state"] = codes
                        print("Using cell_state categorical codes as fallback latent_state")

            # Optional UMAP coordinates.
            if "X_umap" in adata.obsm:
                obs["umap1"] = adata.obsm["X_umap"][:, 0]
                obs["umap2"] = adata.obsm["X_umap"][:, 1]

            # Escape label: use metadata if obvious, otherwise top latent-state quartile.
            escape_text = (
                obs["cell_state"].astype(str)
                + " "
                + obs["condition_or_timepoint"].astype(str)
            ).str.lower()

            escaped_mask = escape_text.str.contains(
                "escape|escaped|relapse|relapsed|resistant|resistance|post|treated|therapy",
                regex=True,
            )

            if escaped_mask.sum() == 0:
                cutoff = obs["latent_state"].quantile(0.75)
                escaped_mask = obs["latent_state"] >= cutoff

            obs["escaped_state"] = np.where(escaped_mask, "escaped", "resident")
            obs["opportunity_score"] = 1.0
            obs["escape_score"] = np.where(obs["escaped_state"] == "escaped", 1.0, 0.0)

            keep = [
                "dataset",
                "sample_id",
                "patient_id",
                "condition_or_timepoint",
                "cell_id",
                "cell_state",
                "latent_state",
                "escaped_state",
                "opportunity_score",
                "escape_score",
            ]

            if "umap1" in obs.columns and "umap2" in obs.columns:
                keep += ["umap1", "umap2"]

            rows.append(obs[keep])

    if not rows:
        raise FileNotFoundError(
            "Could not build Figure6_branching_input_table.csv because no usable "
            "GSE235923/GSE271406 h5ad files were found."
        )

    out = pd.concat(rows, axis=0, ignore_index=True)
    out.to_csv(INPUT, index=False)
    print("Saved:", INPUT)

def standardize_columns(df):
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
    if "ecological_niche" in df.columns and "ecological_context" not in df.columns:
        rename["ecological_niche"] = "ecological_context"

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
        raise ValueError(
            f"Missing required columns: {missing}\n"
            "Minimum required columns are: "
            "dataset, sample_id, patient_id, cell_id, cell_state, latent_state"
        )

    if "condition_or_timepoint" not in df.columns:
        if "timepoint" in df.columns:
            df["condition_or_timepoint"] = df["timepoint"].astype(str)
        elif "condition" in df.columns:
            df["condition_or_timepoint"] = df["condition"].astype(str)
        else:
            df["condition_or_timepoint"] = "unspecified"

    if "escaped_state" not in df.columns:
        cutoff = df["latent_state"].quantile(0.75)
        df["escaped_state"] = np.where(
            df["latent_state"] >= cutoff,
            "escaped",
            "resident",
        )

    if "opportunity_score" not in df.columns:
        df["opportunity_score"] = 1.0

    if "escape_score" not in df.columns:
        df["escape_score"] = np.where(df["escaped_state"].astype(str) == "escaped", 1.0, 0.0)

    if "clone_id" not in df.columns:
        df["clone_id"] = df["cell_state"].astype(str)

    if "ecological_context" not in df.columns:
        df["ecological_context"] = df["cell_state"].astype(str)

    df["latent_state"] = pd.to_numeric(df["latent_state"], errors="coerce")
    df["opportunity_score"] = pd.to_numeric(df["opportunity_score"], errors="coerce").fillna(1.0)
    df["escape_score"] = pd.to_numeric(df["escape_score"], errors="coerce").fillna(0.0)

    return df


def add_embedding_if_missing(df):
    if "umap1" in df.columns and "umap2" in df.columns:
        df["embed1"] = pd.to_numeric(df["umap1"], errors="coerce")
        df["embed2"] = pd.to_numeric(df["umap2"], errors="coerce")
        return df

    features = ["latent_state", "opportunity_score", "escape_score"]

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


def classify_branching_status(df, summary):
    amp_cut = summary["branching_amplification_score"].quantile(0.75)
    amplified_states = set(
        summary.loc[
            summary["branching_amplification_score"] >= amp_cut,
            "branching_unit",
        ]
    )

    df = df.copy()
    df["branching_unit"] = df["dataset"].astype(str) + "|" + df["cell_state"].astype(str)

    df["branching_status"] = "resident"
    df.loc[df["escaped_state"].astype(str) == "escaped", "branching_status"] = "escaped"
    df.loc[df["branching_unit"].isin(amplified_states), "branching_status"] = "amplified"

    return df


def compute_branching_summary(df):
    total_by_dataset = df.groupby("dataset").size().rename("dataset_n_cells")

    rows = []

    group_cols = ["dataset", "cell_state"]

    for (dataset, state), sub in df.groupby(group_cols):
        dataset_n = total_by_dataset.loc[dataset]
        abundance = len(sub) / dataset_n

        resident = df[
            (df["dataset"] == dataset)
            & (df["escaped_state"].astype(str) != "escaped")
        ]

        if len(resident) > 0:
            baseline = resident["latent_state"].mean()
        else:
            baseline = df[df["dataset"] == dataset]["latent_state"].mean()

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

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("branching_amplification_score", ascending=False)

    return summary


def compute_network_edges(df, summary):
    rows = []

    for dataset, dsub in df.groupby("dataset"):
        resident_states = (
            summary[
                (summary["dataset"] == dataset)
                & (summary["escaped_fraction"] < 0.5)
            ]
            .sort_values("abundance", ascending=False)
            .head(3)
        )

        escaped_states = (
            summary[
                (summary["dataset"] == dataset)
                & (summary["escaped_fraction"] >= 0.5)
            ]
            .sort_values("branching_amplification_score", ascending=False)
            .head(6)
        )

        if resident_states.empty:
            resident_states = summary[summary["dataset"] == dataset].head(1)

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

    edges = pd.DataFrame(rows)

    if edges.empty:
        top = summary.head(5)
        for i in range(len(top) - 1):
            rows.append({
                "dataset": top.iloc[i]["dataset"],
                "source": top.iloc[i]["cell_state"],
                "target": top.iloc[i + 1]["cell_state"],
                "weight": top.iloc[i + 1]["branching_amplification_score"],
                "target_abundance": top.iloc[i + 1]["abundance"],
                "target_branching_score": top.iloc[i + 1]["branching_amplification_score"],
            })
        edges = pd.DataFrame(rows)

    return edges


def plot_workflow(ax):
    ax.axis("off")

    xs = [0.12, 0.42, 0.72]
    labels = [
        "Resident\nstate",
        "Escaped\nstate",
        "Branching\namplification",
    ]
    colors = ["#4C78A8", "#F28E2B", "#E15759"]

    for x, label, color in zip(xs, labels, colors):
        circ = plt.Circle((x, 0.55), 0.11, color=color, alpha=0.9, ec="black", lw=1)
        ax.add_patch(circ)
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=8, color="white")

    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow = FancyArrowPatch(
            (x1 + 0.12, 0.55),
            (x2 - 0.12, 0.55),
            arrowstyle="->",
            mutation_scale=14,
            lw=1.8,
            color="black",
        )
        ax.add_patch(arrow)

    ax.text(
        0.42,
        0.22,
        "OU retention → Lévy-like escape → expansion",
        ha="center",
        fontsize=10,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Branching amplification framework")


def plot_embedding(ax, df):
    status_order = ["resident", "escaped", "amplified"]
    for status in status_order:
        sub = df[df["branching_status"] == status]
        if sub.empty:
            continue
        ax.scatter(
            sub["embed1"],
            sub["embed2"],
            s=5,
            alpha=0.65,
            color=STATE_COLORS.get(status, "gray"),
            label=status,
            linewidths=0,
        )

    ax.set_title("Escaped and amplified states")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.legend(frameon=False, markerscale=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_abundance(ax, summary):
    top = summary.head(10).copy()
    top["label"] = top["dataset"].astype(str) + "\n" + top["cell_state"].astype(str)

    colors = sns.color_palette("crest", n_colors=len(top))

    ax.barh(
        top["label"][::-1],
        top["abundance"][::-1],
        color=colors[::-1],
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_title("Expanded state abundance")
    ax.set_xlabel("Fraction of dataset cells")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)


def plot_branching_score(ax, summary):
    top = summary.head(10).copy()
    top["label"] = top["dataset"].astype(str) + "\n" + top["cell_state"].astype(str)

    colors = sns.color_palette("flare", n_colors=len(top))

    ax.barh(
        top["label"][::-1],
        top["branching_amplification_score"][::-1],
        color=colors[::-1],
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_title("Branching-like amplification score")
    ax.set_xlabel("Abundance × displacement × opportunity")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)


def plot_network(ax, edges, summary):
    ax.set_title("State amplification network")

    if edges.empty:
        ax.text(0.5, 0.5, "No network edges available", ha="center", va="center")
        ax.axis("off")
        return

    G = nx.DiGraph()

    for _, row in edges.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])

    node_scores = (
        summary.groupby("cell_state")["branching_amplification_score"]
        .mean()
        .to_dict()
    )

    # Manual layout for clearer publication figure.
    # Put resident/source-like states on the left, intermediate states in the middle,
    # and dominant amplified states on the right.
    nodes = list(G.nodes())
    
    top_scores = (
        summary.groupby("cell_state")["branching_amplification_score"]
        .mean()
        .sort_values(ascending=False)
    )
    
    ranked_nodes = [n for n in top_scores.index if n in nodes]
    
    pos = {}
    
    # Highest amplification state on the upper right.
    if len(ranked_nodes) > 0:
        pos[ranked_nodes[0]] = (1.0, 0.65)
    
    # Next amplified/intermediate states spread vertically in the middle.
    middle_nodes = ranked_nodes[1:5]
    middle_y = np.linspace(0.8, 0.2, max(len(middle_nodes), 1))
    
    for n, y in zip(middle_nodes, middle_y):
        pos[n] = (0.45, y)
    
    # Remaining states on the left/lower-left.
    remaining = [n for n in nodes if n not in pos]
    remaining_y = np.linspace(0.85, 0.15, max(len(remaining), 1))
    
    for n, y in zip(remaining, remaining_y):
        pos[n] = (0.05, y)
    
    # Slightly separate common auxiliary/other states if present.
    if "other_state" in pos:
        pos["other_state"] = (0.05, 0.12)
    if "not_malignant" in pos:
        pos["not_malignant"] = (0.45, 0.35)

    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    if len(weights) > 0:
        widths = 0.8 + 4.0 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    else:
        widths = [1.0]

    node_values = np.array([node_scores.get(n, 0.0) for n in G.nodes()])
    node_sizes = 400 + 1300 * (
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

    label_pos = {}
    
    for n, (x, y) in pos.items():
        if n == "state_MonoDC":
            label_pos[n] = (x, y + 0.11)
        else:
            label_pos[n] = (x, y + 0.055)
    
    nx.draw_networkx_labels(
        G,
        label_pos,
        ax=ax,
        font_size=7,
        font_weight="normal",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            pad=0.8,
        ),
    )
    
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0.0, 1.0)

    ax.axis("off")


def plot_conceptual(ax):
    ax.axis("off")

    x = np.linspace(-2.5, 2.5, 300)
    U = 0.45 * (x + 1.0) ** 2
    ax.plot(x, U, color="#4C78A8", lw=2)

    ax.scatter([-1.0], [0], color="#4C78A8", s=80, zorder=3)
    ax.text(-1.0, -0.3, "OU-like\nretention", ha="center", va="top", fontsize=9)

    ax.annotate(
        "",
        xy=(2.4, 1.0),
        xytext=(0.1, 0.5),
        arrowprops=dict(
            arrowstyle="->",
            linestyle="--",
            lw=1.8,
            color="black",
            connectionstyle="arc3,rad=-0.25",
        ),
    )

    centers = [(3.7, 1.0), (4.3, 1.35), (4.6, 0.7), (5.1, 1.2), (5.4, 0.85)]
    sizes = [70, 95, 120, 160, 210]

    for (cx, cy), s in zip(centers, sizes):
        ax.scatter([cx], [cy], s=s, color="#E15759", alpha=0.85, edgecolor="black", lw=0.5)

    ax.text(2.1, 1.55, "Lévy-like\nescape", ha="center", fontsize=9)
    ax.text(4.7, 1.75, "branching-like\namplification", ha="center", fontsize=9)

    ax.set_xlim(-2.8, 6.0)
    ax.set_ylim(-0.6, 3.0)
    ax.set_title("OU-Lévy-branching interpretation")


def main():
    build_input_table_from_h5ad_if_missing()

    df = pd.read_csv(INPUT)
    df = standardize_columns(df)
    df = df.dropna(subset=["latent_state"]).copy()
    df = add_embedding_if_missing(df)

    summary = compute_branching_summary(df)
    df = classify_branching_status(df, summary)
    edges = compute_network_edges(df, summary)

    summary.to_csv(SUMMARY_CSV, index=False)
    edges.to_csv(EDGE_CSV, index=False)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.1, 1.1],
        height_ratios=[1, 1],
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    plot_workflow(axA)
    plot_embedding(axB, df)
    plot_abundance(axC, summary)
    plot_branching_score(axD, summary)
    plot_network(axE, edges, summary)
    plot_conceptual(axF)

    for ax, label in zip([axA, axB, axC, axD, axE, axF], list("ABCDEF")):
        ax.text(
            -0.12,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="bottom",
            ha="right",
            clip_on=False,
        )

    fig.suptitle(
        "Branching-like amplification of escaped ecological states",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.35, hspace=0.35)

    fig.savefig(FIG_PNG, dpi=600)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)

    print("Saved Figure 6 outputs to:", OUT)
    print("Input table:", INPUT)
    print("Summary table:", SUMMARY_CSV)
    print("Network edge table:", EDGE_CSV)


if __name__ == "__main__":
    main()
