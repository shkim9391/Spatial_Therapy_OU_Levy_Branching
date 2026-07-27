from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from matplotlib.patches import FancyArrowPatch

BASE = Path("/Spatial_Therapy_OU_Levy_Branching")
OUT = BASE / "Figure_7"
OUT.mkdir(exist_ok=True)

INPUT = OUT / "Figure7_therapy_response_input_table.csv"
RESPONSE_META = OUT / "Figure7_response_metadata.csv"

FIG6_SUMMARY = BASE / "Figure_6" / "Figure6_branching_amplification_summary.csv"
FIG5_ESCAPE = BASE / "Figure_5" / "Figure5_context_escape_scores.csv"
FIG4_OU = BASE / "Figure_4" / "Figure4_spatial_ou_parameter_summary.csv"

FIG_PNG = OUT / "Figure7_therapy_response_validation.png"
FIG_PDF = OUT / "Figure7_therapy_response_validation.pdf"
FIG_SVG = OUT / "Figure7_therapy_response_validation.svg"
SUMMARY_CSV = OUT / "Figure7_therapy_response_summary.csv"

RESPONSE_ORDER = [
    "responder",
    "persistent_disease",
    "relapse-associated",
    "unknown",
]

RESPONSE_COLORS = {
    "responder": "#4C78A8",
    "persistent_disease": "#F28E2B",
    "relapse-associated": "#E15759",
    "unknown": "#B0B0B0",
}


def infer_response_status(text):
    text = str(text).lower()

    if any(k in text for k in ["relapse-associated", "relapse", "relapsed", "rel", "resistant", "refractory"]):
        return "relapse-associated"

    if any(k in text for k in ["persistent", "mrd", "residual", "post", "treated", "therapy"]):
        return "persistent_disease"

    if any(k in text for k in ["responder", "response", "remission", "cr", "complete"]):
        return "responder"

    return "unknown"


def build_input_table_if_missing():
    if INPUT.exists():
        return

    print("Building draft Figure 7 input table:", INPUT)

    rows = []

    if FIG6_SUMMARY.exists():
        fig6 = pd.read_csv(FIG6_SUMMARY)

        for _, r in fig6.iterrows():
            dataset = r.get("dataset", "GSE235923")
            cell_state = r.get("cell_state", "state")
            sample_id = r.get("sample_id", cell_state)

            rows.append({
                "dataset": dataset,
                "patient_id": str(sample_id),
                "sample_id": str(sample_id),
                "response_status": infer_response_status(str(sample_id) + " " + str(cell_state)),
                "ou_retention_score": np.nan,
                "levy_escape_score": np.nan,
                "branching_amplification_score": r.get("branching_amplification_score", np.nan),
                "cell_state": cell_state,
            })

    if not rows:
        raise FileNotFoundError(
            f"Input table not found: {INPUT}\n"
            "Create Figure7_therapy_response_input_table.csv with columns:\n"
            "dataset, patient_id, sample_id, response_status, "
            "ou_retention_score, levy_escape_score, branching_amplification_score"
        )

    df = pd.DataFrame(rows)

    # Optional response metadata override.
    if RESPONSE_META.exists():
        meta = pd.read_csv(RESPONSE_META)
        merge_cols = [c for c in ["dataset", "sample_id", "patient_id"] if c in meta.columns and c in df.columns]

        if "response_status" in meta.columns and merge_cols:
            df = df.drop(columns=["response_status"], errors="ignore").merge(
                meta[merge_cols + ["response_status"]].drop_duplicates(),
                on=merge_cols,
                how="left",
            )
            df["response_status"] = df["response_status"].fillna("unknown")

    # If OU/Lévy values are unavailable, use dataset-level placeholders from available summaries.
    if df["ou_retention_score"].isna().all() and FIG4_OU.exists():
        ou = pd.read_csv(FIG4_OU)
        if "theta" in ou.columns:
            df["ou_retention_score"] = np.nanmean(ou["theta"])

    if df["levy_escape_score"].isna().all() and FIG5_ESCAPE.exists():
        esc = pd.read_csv(FIG5_ESCAPE)
        if "levy_like_escape_score" in esc.columns:
            df["levy_escape_score"] = np.nanmean(esc["levy_like_escape_score"])

    df["ou_retention_score"] = df["ou_retention_score"].fillna(0.0)
    df["levy_escape_score"] = df["levy_escape_score"].fillna(0.0)
    df["branching_amplification_score"] = df["branching_amplification_score"].fillna(0.0)

    df.to_csv(INPUT, index=False)
    print("Saved:", INPUT)


def standardize_input(df):
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
        raise ValueError(f"Missing required columns: {missing}")

    for c in ["ou_retention_score", "levy_escape_score", "branching_amplification_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["response_status"] = df["response_status"].astype(str).str.lower()
    df["response_status"] = df["response_status"].map(infer_response_status)

    return df


def add_integrated_score(df):
    score_cols = [
        "ou_retention_score",
        "levy_escape_score",
        "branching_amplification_score",
    ]

    z = df[score_cols].apply(lambda x: zscore(x, nan_policy="omit") if x.std() > 0 else np.zeros(len(x)))
    df["integrated_olb_risk_score"] = z.sum(axis=1)

    return df


def plot_workflow(ax):
    ax.axis("off")

    xs = [0.12, 0.38, 0.64, 0.88]
    labels = [
        "Baseline\necology",
        "Therapy\npressure",
        "Residual\npersistence",
        "Relapse / resistant\nexpansion",
    ]
    colors = ["#4C78A8", "#9E9E9E", "#F28E2B", "#E15759"]

    for x, label, color in zip(xs, labels, colors):
        circ = plt.Circle((x, 0.58), 0.095, color=color, alpha=0.9, ec="black", lw=1)
        ax.add_patch(circ)
        ax.text(x, 0.58, label, ha="center", va="center", fontsize=8, color="white")

    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow = FancyArrowPatch(
            (x1 + 0.1, 0.58),
            (x2 - 0.1, 0.58),
            arrowstyle="->",
            mutation_scale=13,
            lw=1.6,
            color="black",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.25,
        "OU retention + Lévy-like escape + branching-like amplification",
        ha="center",
        fontsize=9,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Relapse-associated validation framework")


def plot_response_distribution(ax, df):
    counts = (
        df["response_status"]
        .value_counts()
        .reindex(RESPONSE_ORDER)
        .fillna(0)
    )

    colors = [RESPONSE_COLORS[r] for r in counts.index]

    ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Relapse-associated validation categories")
    ax.set_xlabel("Response status")
    ax.set_ylabel("Number of samples/states")
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)


def plot_metric_by_response(ax, df, metric, title, ylabel):
    order = [r for r in RESPONSE_ORDER if r in df["response_status"].unique()]
    palette = {r: RESPONSE_COLORS[r] for r in order}

    sns.boxplot(
        data=df,
        x="response_status",
        y=metric,
        order=order,
        palette=palette,
        showfliers=False,
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

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)


def plot_integrated_heatmap(ax, df):
    show_cols = [
        "ou_retention_score",
        "levy_escape_score",
        "branching_amplification_score",
        "integrated_olb_risk_score",
    ]

    plot_df = df.copy()
    plot_df["label"] = plot_df["dataset"].astype(str) + "\n" + plot_df["sample_id"].astype(str)

    plot_df = plot_df.sort_values("integrated_olb_risk_score", ascending=False).head(12)

    X = plot_df[show_cols].copy()
    X = X.apply(lambda x: zscore(x, nan_policy="omit") if x.std() > 0 else np.zeros(len(x)))

    sns.heatmap(
        X,
        cmap="vlag",
        center=0,
        yticklabels=plot_df["label"],
        xticklabels=[
            "OU\nretention",
            "Lévy\nescape",
            "Branching\namplification",
            "Integrated\nrisk",
        ],
        ax=ax,
        cbar_kws={"label": "z-score"},
    )

    ax.set_title("Integrated OU-Lévy-branching risk profile")
    ax.set_xlabel("")
    ax.set_ylabel("")


def main():
    build_input_table_if_missing()

    df = pd.read_csv(INPUT)
    df = standardize_input(df)
    df = add_integrated_score(df)

    df.to_csv(SUMMARY_CSV, index=False)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.15], height_ratios=[1, 1])

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    plot_workflow(axA)
    plot_response_distribution(axB, df)

    plot_metric_by_response(
        axC,
        df,
        "ou_retention_score",
        "OU-like retention by relapse association",
        "OU retention score",
    )

    plot_metric_by_response(
        axD,
        df,
        "levy_escape_score",
        "Lévy-like escape by relapse association",
        "Lévy-like escape score",
    )

    plot_metric_by_response(
        axE,
        df,
        "branching_amplification_score",
        "Branching-like amplification by relapse association",
        "Branching amplification score",
    )

    plot_integrated_heatmap(axF, df)

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
        "Relapse-associated validation of OU-Lévy-branching ecological summaries",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.35, hspace=0.42)

    fig.savefig(FIG_PNG, dpi=600)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)

    print("Saved Figure 7 outputs to:", OUT)
    print("Input table:", INPUT)
    print("Summary table:", SUMMARY_CSV)


if __name__ == "__main__":
    main()
