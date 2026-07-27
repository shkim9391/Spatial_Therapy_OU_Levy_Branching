from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import networkx as nx

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
IN = BASE / "processed" / "spatial_ecological_contexts"
OUT = BASE / "processed" / "evolutionary_landscape"
FIG = BASE / "figures" / "evolutionary_landscape"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

assign = pd.read_csv(IN / "GSE279576_spatial_ecological_context_assignments.csv")
means_z = pd.read_csv(IN / "GSE279576_spatial_ecological_context_feature_means_z.csv", index_col=0)

# -----------------------------
# 1. Context opportunity score
# -----------------------------
positive = [
    "AML_blast_like_score",
    "Primitive_like_AML_score",
    "CXCL12_CXCR4_axis_score",
    "MSC_stromal_score",
    "Hypoxia_stress_score",
    "Immune_suppression_score",
]

negative = [
    "T_NK_score",
]

available_pos = [x for x in positive if x in means_z.columns]
available_neg = [x for x in negative if x in means_z.columns]

opp = means_z[available_pos].mean(axis=1) - means_z[available_neg].mean(axis=1)
opp = (opp - opp.mean()) / opp.std()

opportunity = pd.DataFrame({
    "context": means_z.index,
    "evolutionary_opportunity_z": opp
}).set_index("context")

opportunity.to_csv(OUT / "GSE279576_context_evolutionary_opportunity.csv")

# -----------------------------
# 2. Context distance/similarity
# -----------------------------
D = pairwise_distances(means_z.values, metric="euclidean")
contexts = list(means_z.index)

dist_df = pd.DataFrame(D, index=contexts, columns=contexts)
dist_df.to_csv(OUT / "GSE279576_context_ecological_distance.csv")

sigma = np.median(D[D > 0])
S = np.exp(-(D ** 2) / (2 * sigma ** 2))
np.fill_diagonal(S, 0)

sim_df = pd.DataFrame(S, index=contexts, columns=contexts)
sim_df.to_csv(OUT / "GSE279576_context_ecological_similarity.csv")

# -----------------------------
# 3. Context transition proxy
# -----------------------------
# Similarity weighted by opportunity of destination context.
T = S * np.exp(opportunity.loc[contexts, "evolutionary_opportunity_z"].values[None, :])
T = T / T.sum(axis=1, keepdims=True)

trans_df = pd.DataFrame(T, index=contexts, columns=contexts)
trans_df.to_csv(OUT / "GSE279576_context_transition_proxy.csv")

# -----------------------------
# 4. Plot network
# -----------------------------
G = nx.DiGraph()

freq = assign["spatial_ecological_context"].value_counts(normalize=True).to_dict()

for ctx in contexts:
    G.add_node(
        ctx,
        opportunity=float(opportunity.loc[ctx, "evolutionary_opportunity_z"]),
        freq=float(freq.get(ctx, 0))
    )

for i, a in enumerate(contexts):
    for j, b in enumerate(contexts):
        if a == b:
            continue
        if T[i, j] > 0.12:
            G.add_edge(a, b, weight=float(T[i, j]))

pos = nx.spring_layout(G, seed=7, weight="weight")

fig, ax = plt.subplots(figsize=(7, 6))

node_sizes = [2500 * G.nodes[n]["freq"] + 700 for n in G.nodes()]
node_colors = [G.nodes[n]["opportunity"] for n in G.nodes()]

nx.draw_networkx_edges(
    G, pos, ax=ax,
    width=[6 * G[u][v]["weight"] for u, v in G.edges()],
    alpha=0.45,
    arrows=True,
    arrowsize=18,
    connectionstyle="arc3,rad=0.08"
)

nodes = nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_size=node_sizes,
    node_color=node_colors,
    cmap="viridis",
    edgecolors="black",
    linewidths=1.2
)

nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight="bold")

cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Evolutionary opportunity z-score")

ax.set_title("Context-level evolutionary landscape", fontsize=14)
ax.axis("off")
plt.tight_layout()

plt.savefig(FIG / "Figure3_context_evolutionary_network.png", dpi=300)
plt.savefig(FIG / "Figure3_context_evolutionary_network.pdf")
plt.close()

# -----------------------------
# 5. Opportunity bar plot
# -----------------------------
opp_sorted = opportunity.sort_values("evolutionary_opportunity_z", ascending=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(opp_sorted.index, opp_sorted["evolutionary_opportunity_z"])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Evolutionary opportunity z-score")
ax.set_xlabel("Spatial ecological context")
ax.set_title("Evolutionary opportunity by spatial context")
plt.tight_layout()

plt.savefig(FIG / "Figure3_context_opportunity_scores.png", dpi=300)
plt.savefig(FIG / "Figure3_context_opportunity_scores.pdf")
plt.close()

print("Saved outputs to:")
print(OUT)
print(FIG)

print("\nEvolutionary opportunity:")
print(opportunity.sort_values("evolutionary_opportunity_z", ascending=False))
