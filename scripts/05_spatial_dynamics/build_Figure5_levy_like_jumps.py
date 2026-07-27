from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
OUT = Path("/Spatial_Therapy_OU_Levy_Branching/Figure_5")
OUT.mkdir(exist_ok=True)

CTX_DIR = BASE / "processed" / "spatial_ecological_contexts"
LAND_DIR = BASE / "processed" / "evolutionary_landscape"

ASSIGN_FILE = CTX_DIR / "GSE279576_spatial_ecological_context_assignments.csv"
FEATURE_FILE = CTX_DIR / "GSE279576_spatial_ecological_context_feature_means_z.csv"
OPP_FILE = LAND_DIR / "GSE279576_context_evolutionary_opportunity.csv"
OU_INPUT_FILE = BASE / "processed" / "spatial_ou_input_table.csv"

NICHE_ORDER = ["S1", "S2", "S3", "S4", "S5"]
NICHE_COLORS = {
    "S1": "#4C78A8",
    "S2": "#59A14F",
    "S3": "#F28E2B",
    "S4": "#E15759",
    "S5": "#B279A2",
}

K_NEIGHBORS = 6
RARE_JUMP_QUANTILE = 0.90


def load_inputs():
    assign = pd.read_csv(ASSIGN_FILE)

    rename = {}
    if "spatial_x" in assign.columns and "x" not in assign.columns:
        rename["spatial_x"] = "x"
    if "spatial_y" in assign.columns and "y" not in assign.columns:
        rename["spatial_y"] = "y"
    if "spatial_ecological_context" in assign.columns and "ecological_context" not in assign.columns:
        rename["spatial_ecological_context"] = "ecological_context"
    if "spot_barcode" in assign.columns and "spot_id" not in assign.columns:
        rename["spot_barcode"] = "spot_id"

    assign = assign.rename(columns=rename)

    required = ["sample_id", "spot_id", "x", "y", "ecological_context"]
    missing = [c for c in required if c not in assign.columns]
    if missing:
        raise ValueError(f"Missing required columns in assignment file: {missing}")

    if OU_INPUT_FILE.exists():
        ou = pd.read_csv(OU_INPUT_FILE)
        ou_rename = {}
        if "ecological_niche" in ou.columns and "ecological_context" not in ou.columns:
            ou_rename["ecological_niche"] = "ecological_context"
        if "latent_state" in ou.columns:
            pass
        ou = ou.rename(columns=ou_rename)

        merge_cols = ["sample_id", "spot_id"]
        if all(c in ou.columns for c in merge_cols + ["latent_state"]):
            assign = assign.merge(
                ou[merge_cols + ["latent_state"]],
                on=merge_cols,
                how="left",
            )

    feature_means_z = pd.read_csv(FEATURE_FILE, index_col=0)
    feature_means_z = feature_means_z.loc[
        [c for c in NICHE_ORDER if c in feature_means_z.index]
    ]

    opp = pd.read_csv(OPP_FILE, index_col=0)
    if "evolutionary_opportunity_z" in opp.columns:
        opp = opp["evolutionary_opportunity_z"]
    else:
        opp = opp.iloc[:, 0]

    return assign, feature_means_z, opp


def compute_context_ecological_distances(feature_means_z):
    """
    Ecological distance between S1-S5 contexts is computed from standardized
    context-level feature profiles.
    """
    contexts = feature_means_z.index.tolist()
    X = feature_means_z.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    dist = squareform(pdist(X, metric="euclidean"))
    dist_df = pd.DataFrame(dist, index=contexts, columns=contexts)

    # 2D layout for niche landscape plotting.
    if X.shape[0] >= 2:
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(X)
    else:
        coords = np.zeros((X.shape[0], 2))

    coord_df = pd.DataFrame(coords, index=contexts, columns=["EcoPC1", "EcoPC2"])

    return dist_df, coord_df


def build_neighbor_edges(df, context_dist):
    """
    Build spatial k-nearest-neighbor edges within each sample and annotate
    same-context retention, cross-context transitions, ecological jump distance,
    and latent-state displacement.
    """
    all_edges = []

    for sample_id, sub in df.groupby("sample_id"):
        sub = sub.dropna(subset=["x", "y", "ecological_context"]).copy()
        sub = sub[sub["ecological_context"].isin(context_dist.index)]

        if len(sub) <= K_NEIGHBORS:
            continue

        coords = sub[["x", "y"]].to_numpy(dtype=float)

        nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)

        sub = sub.reset_index(drop=True)

        seen = set()

        for i in range(len(sub)):
            for rank in range(1, K_NEIGHBORS + 1):
                j = indices[i, rank]

                a = min(i, j)
                b = max(i, j)
                edge_key = (sample_id, a, b)

                if edge_key in seen:
                    continue
                seen.add(edge_key)

                row_i = sub.iloc[i]
                row_j = sub.iloc[j]

                ctx_i = row_i["ecological_context"]
                ctx_j = row_j["ecological_context"]

                eco_dist = context_dist.loc[ctx_i, ctx_j]
                same_context = ctx_i == ctx_j
                cross_context = not same_context

                latent_delta = np.nan
                if "latent_state" in sub.columns:
                    if pd.notna(row_i.get("latent_state")) and pd.notna(row_j.get("latent_state")):
                        latent_delta = abs(row_i["latent_state"] - row_j["latent_state"])

                all_edges.append({
                    "sample_id": sample_id,
                    "spot_i": row_i["spot_id"],
                    "spot_j": row_j["spot_id"],
                    "x_i": row_i["x"],
                    "y_i": row_i["y"],
                    "x_j": row_j["x"],
                    "y_j": row_j["y"],
                    "context_i": ctx_i,
                    "context_j": ctx_j,
                    "spatial_distance": distances[i, rank],
                    "ecological_distance": eco_dist,
                    "same_context": same_context,
                    "cross_context": cross_context,
                    "latent_delta": latent_delta,
                })

    edges = pd.DataFrame(all_edges)

    if edges.empty:
        raise ValueError("No neighbor edges were constructed.")

    cross = edges[edges["cross_context"]].copy()
    if len(cross) > 0:
        threshold = cross["ecological_distance"].quantile(RARE_JUMP_QUANTILE)
    else:
        threshold = np.nan

    edges["rare_jump"] = (
        edges["cross_context"]
        & (edges["ecological_distance"] >= threshold)
    )

    return edges, threshold


def compute_escape_scores(edges):
    """
    Context-level Lévy-like escape score:
    mean ecological distance among outgoing cross-context neighbor edges,
    multiplied by cross-context transition frequency.
    """
    rows = []

    directed = []

    for _, e in edges.iterrows():
        directed.append({
            "sample_id": e["sample_id"],
            "source_context": e["context_i"],
            "target_context": e["context_j"],
            "ecological_distance": e["ecological_distance"],
            "cross_context": e["cross_context"],
            "rare_jump": e["rare_jump"],
        })
        directed.append({
            "sample_id": e["sample_id"],
            "source_context": e["context_j"],
            "target_context": e["context_i"],
            "ecological_distance": e["ecological_distance"],
            "cross_context": e["cross_context"],
            "rare_jump": e["rare_jump"],
        })

    directed = pd.DataFrame(directed)

    for ctx in NICHE_ORDER:
        sub = directed[directed["source_context"] == ctx]
        if sub.empty:
            continue

        cross = sub[sub["cross_context"]]
        rare = sub[sub["rare_jump"]]

        cross_frequency = len(cross) / len(sub)
        rare_frequency = len(rare) / len(sub)

        mean_jump_distance = cross["ecological_distance"].mean() if len(cross) else 0.0
        mean_rare_distance = rare["ecological_distance"].mean() if len(rare) else 0.0

        escape_score = cross_frequency * mean_jump_distance
        rare_escape_score = rare_frequency * mean_rare_distance

        rows.append({
            "ecological_context": ctx,
            "n_edges": len(sub),
            "cross_frequency": cross_frequency,
            "rare_frequency": rare_frequency,
            "mean_jump_distance": mean_jump_distance,
            "mean_rare_distance": mean_rare_distance,
            "levy_like_escape_score": escape_score,
            "rare_escape_score": rare_escape_score,
        })

    return pd.DataFrame(rows)


def choose_example_sample(df):
    preferred = ["GSM8576301_BM1", "GSM8576303_BM2", "GSM8576306_EM2_v1"]
    available = df["sample_id"].dropna().unique().tolist()
    return next((s for s in preferred if s in available), sorted(available)[0])


def plot_context_landscape(ax, context_coords, opp, context_dist):
    contexts = [c for c in NICHE_ORDER if c in context_coords.index]

    G = nx.Graph()
    for c in contexts:
        G.add_node(c)

    # Draw all pairwise ecological proximities, with closer contexts as stronger edges.
    vals = []
    for i, c1 in enumerate(contexts):
        for c2 in contexts[i + 1:]:
            d = context_dist.loc[c1, c2]
            vals.append(d)

    vals = np.array(vals)
    d_min = vals.min()
    d_max = vals.max()

    for i, c1 in enumerate(contexts):
        for c2 in contexts[i + 1:]:
            d = context_dist.loc[c1, c2]
            proximity = 1.0 - (d - d_min) / (d_max - d_min + 1e-9)
            G.add_edge(c1, c2, weight=proximity, distance=d)

    pos = {
        c: (context_coords.loc[c, "EcoPC1"], context_coords.loc[c, "EcoPC2"])
        for c in contexts
    }

    edge_widths = [0.5 + 3.0 * G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        alpha=0.35,
        edge_color="gray",
    )

    node_values = [opp.get(c, np.nan) for c in contexts]
    vmin = np.nanmin(node_values)
    vmax = np.nanmax(node_values)

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_values,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        node_size=900,
        edgecolors="black",
        linewidths=1.2,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        labels={c: c for c in contexts},
        font_size=10,
        font_weight="bold",
    )

    ax.set_title("Spatial niche landscape")
    ax.set_xlabel("Context EcoPC1")
    ax.set_ylabel("Context EcoPC2")
    ax.spines[["top", "right"]].set_visible(False)

    cbar = plt.colorbar(
        nodes,
        ax=ax,
        orientation="horizontal",
        fraction=0.055,
        pad=0.12,
    )
    cbar.set_label("Evolutionary opportunity", fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def plot_neighbor_transition_summary(ax, edges):
    pair_counts = (
        edges[edges["cross_context"]]
        .assign(pair=lambda x: x.apply(
            lambda r: "-".join(sorted([r["context_i"], r["context_j"]])),
            axis=1,
        ))
        .groupby("pair")
        .size()
        .sort_values(ascending=False)
    )

    if pair_counts.empty:
        ax.text(0.5, 0.5, "No cross-context edges", ha="center", va="center")
        ax.axis("off")
        return

    pair_counts = pair_counts.head(10)

    ax.barh(
        pair_counts.index[::-1],
        pair_counts.values[::-1],
        color="#4C78A8",
        alpha=0.85,
    )

    ax.set_title("Cross-context neighbor transitions")
    ax.set_xlabel("Number of spatial neighbor edges")
    ax.set_ylabel("Context pair")
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def plot_jump_distribution(ax, edges, rare_threshold):
    cross = edges[edges["cross_context"]].copy()

    if cross.empty:
        ax.text(0.5, 0.5, "No cross-context jumps", ha="center", va="center")
        ax.axis("off")
        return

    sns.histplot(
        data=cross,
        x="ecological_distance",
        bins=18,
        color="#7B3294",
        alpha=0.75,
        ax=ax,
    )

    if np.isfinite(rare_threshold):
        ax.axvline(
            rare_threshold,
            color="black",
            linestyle="--",
            lw=1.5,
            label=f"Rare-jump cutoff\nq={RARE_JUMP_QUANTILE:.2f}",
        )
        ax.legend(frameon=False, fontsize=8)

    ax.set_title("Ecological jump-distance distribution")
    ax.set_xlabel("Ecological distance between contexts")
    ax.set_ylabel("Cross-context edge count")
    ax.spines[["top", "right"]].set_visible(False)


def plot_escape_scores(ax, escape_df):
    escape_df = escape_df.copy()
    escape_df["ecological_context"] = pd.Categorical(
        escape_df["ecological_context"],
        categories=NICHE_ORDER,
        ordered=True,
    )
    escape_df = escape_df.sort_values("ecological_context")

    colors = [NICHE_COLORS.get(c, "gray") for c in escape_df["ecological_context"]]

    ax.bar(
        escape_df["ecological_context"],
        escape_df["levy_like_escape_score"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )

    ax.set_title("Lévy-like escape score by context")
    ax.set_xlabel("Spatial ecological context")
    ax.set_ylabel("Cross-context frequency ×\necological distance")
    ax.spines[["top", "right"]].set_visible(False)


def plot_spatial_rare_jump_map(ax, df, edges, example_sample):
    sub = df[df["sample_id"] == example_sample].copy()
    edge_sub = edges[edges["sample_id"] == example_sample].copy()

    for ctx in NICHE_ORDER:
        s = sub[sub["ecological_context"] == ctx]
        if s.empty:
            continue
        ax.scatter(
            s["x"],
            s["y"],
            s=7,
            color=NICHE_COLORS.get(ctx, "gray"),
            alpha=0.75,
            linewidths=0,
            label=ctx,
        )

    local_edges = edge_sub[
        edge_sub["cross_context"] & (~edge_sub["rare_jump"])
    ].sample(
        n=min(250, max(1, len(edge_sub[edge_sub["cross_context"] & (~edge_sub["rare_jump"])]))),
        random_state=0,
        replace=False,
    ) if len(edge_sub[edge_sub["cross_context"] & (~edge_sub["rare_jump"])]) > 0 else pd.DataFrame()

    rare_edges = edge_sub[edge_sub["rare_jump"]]

    for _, e in local_edges.iterrows():
        ax.plot(
            [e["x_i"], e["x_j"]],
            [e["y_i"], e["y_j"]],
            color="gray",
            alpha=0.15,
            lw=0.5,
        )

    for _, e in rare_edges.iterrows():
        ax.plot(
            [e["x_i"], e["x_j"]],
            [e["y_i"], e["y_j"]],
            color="black",
            alpha=0.85,
            lw=1.2,
        )

    ax.set_title(f"{example_sample}: rare ecological jumps")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        markerscale=2,
    )


def plot_conceptual_panel(ax):
    ax.axis("off")

    basin_x = np.linspace(-2.5, 2.5, 300)
    U1 = 0.4 * (basin_x + 1.1) ** 2
    U2 = 0.65 * (basin_x - 1.2) ** 2 + 0.4

    ax.plot(basin_x, U1, color="#4C78A8", lw=2)
    ax.plot(basin_x + 5.5, U2, color="#B279A2", lw=2)

    ax.scatter([-1.1], [0], color="#4C78A8", s=80, zorder=3)
    ax.scatter([6.7], [0.4], color="#B279A2", s=80, zorder=3)

    ax.annotate(
        "",
        xy=(5.6, 0.7),
        xytext=(1.1, 0.4),
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
            linestyle="--",
            color="black",
            connectionstyle="arc3,rad=-0.35",
        ),
    )

    ax.text(-1.1, -0.35, "OU retention\nwithin context", ha="center", va="top", fontsize=9)
    ax.text(3.2, 2.1, "rare Lévy-like\ncross-context jump", ha="center", fontsize=9)
    ax.text(6.7, -0.05, "new ecological\nbasin", ha="center", va="top", fontsize=9)

    ax.set_xlim(-3.0, 8.5)
    ax.set_ylim(-0.7, 3.2)
    ax.set_title("OU retention plus rare ecological escape")


def main():
    df, feature_means_z, opp = load_inputs()

    df = df[df["ecological_context"].isin(NICHE_ORDER)].copy()
    df["ecological_context"] = pd.Categorical(
        df["ecological_context"],
        categories=NICHE_ORDER,
        ordered=True,
    )

    context_dist, context_coords = compute_context_ecological_distances(feature_means_z)
    edges, rare_threshold = build_neighbor_edges(df, context_dist)
    escape_df = compute_escape_scores(edges)

    edges.to_csv(OUT / "Figure5_neighbor_edge_table.csv", index=False)
    escape_df.to_csv(OUT / "Figure5_context_escape_scores.csv", index=False)
    context_dist.to_csv(OUT / "Figure5_context_ecological_distance_matrix.csv")

    example_sample = choose_example_sample(df)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.05, 1.15],
        height_ratios=[1, 1],
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    plot_context_landscape(axA, context_coords, opp, context_dist)
    plot_neighbor_transition_summary(axB, edges)
    plot_jump_distribution(axC, edges, rare_threshold)
    plot_escape_scores(axD, escape_df)
    plot_spatial_rare_jump_map(axE, df, edges, example_sample)
    plot_conceptual_panel(axF)

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

#    fig.suptitle(
#        "Lévy-like ecological escape across the spatial niche landscape",
#        fontsize=16,
#        fontweight="bold",
#    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.45, hspace=0.35)

    fig.savefig(OUT / "Figure5_levy_like_ecological_escape.png", dpi=600)
    fig.savefig(OUT / "Figure5_levy_like_ecological_escape.pdf")
    fig.savefig(OUT / "Figure5_levy_like_ecological_escape.svg")
    plt.close(fig)

    print("Saved Figure 5 outputs to:", OUT)
    print("Representative sample:", example_sample)
    print("Rare-jump ecological-distance cutoff:", rare_threshold)


if __name__ == "__main__":
    main()
