from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("/Spatial_Therapy_OU_Levy_Branching/GSE279576")
IN = BASE / "processed" / "spatial_ecological_contexts"
OUT = IN / "context_labels"
OUT.mkdir(parents=True, exist_ok=True)

assign = pd.read_csv(IN / "GSE279576_spatial_ecological_context_assignments.csv")
means_z = pd.read_csv(IN / "GSE279576_spatial_ecological_context_feature_means_z.csv", index_col=0)
means = pd.read_csv(IN / "GSE279576_spatial_ecological_context_feature_means.csv", index_col=0)

# Count and fraction by sample
count_by_sample = pd.crosstab(assign["sample_id"], assign["spatial_ecological_context"])
frac_by_sample = pd.crosstab(
    assign["sample_id"],
    assign["spatial_ecological_context"],
    normalize="index"
)

# Count and fraction by site
count_by_site = pd.crosstab(assign["site"], assign["spatial_ecological_context"])
frac_by_site = pd.crosstab(
    assign["site"],
    assign["spatial_ecological_context"],
    normalize="index"
)

# Enrichment ratios: context fraction in site / global context fraction
global_frac = assign["spatial_ecological_context"].value_counts(normalize=True).sort_index()
site_enrichment = frac_by_site.div(global_frac, axis=1)

# Top defining features per context
top_rows = []
for ctx in means_z.index:
    s = means_z.loc[ctx].sort_values(ascending=False)
    top_rows.append({
        "context": ctx,
        "top_positive_features": "; ".join([f"{k}={v:.2f}" for k, v in s.head(6).items()]),
        "top_negative_features": "; ".join([f"{k}={v:.2f}" for k, v in s.tail(4).items()]),
    })

top_features = pd.DataFrame(top_rows)

# Manual biological labels based on current heatmap
label_map = {
    "S1": "blast–committed myeloid / hypoxia-stress context",
    "S2": "primitive AML / HSPC-like context",
    "S3": "immune-suppressed monocyte–macrophage / B-cell context",
    "S4": "inflammatory stromal–vascular / ECM niche context",
    "S5": "T/NK immune-active context",
}

context_labels = pd.DataFrame({
    "context": sorted(assign["spatial_ecological_context"].unique()),
})
context_labels["proposed_label"] = context_labels["context"].map(label_map)

context_labels = context_labels.merge(top_features, on="context", how="left")

# Add BM/EM enrichment
bm_frac = frac_by_site.loc["BM"] if "BM" in frac_by_site.index else pd.Series(dtype=float)
em_frac = frac_by_site.loc["EM"] if "EM" in frac_by_site.index else pd.Series(dtype=float)

context_labels["BM_fraction"] = context_labels["context"].map(bm_frac)
context_labels["EM_fraction"] = context_labels["context"].map(em_frac)
context_labels["BM_to_EM_ratio"] = (
    (context_labels["BM_fraction"] + 1e-6) /
    (context_labels["EM_fraction"] + 1e-6)
)

def site_bias(row):
    if row["BM_to_EM_ratio"] > 1.5:
        return "BM-enriched"
    if row["BM_to_EM_ratio"] < 0.67:
        return "EM-enriched"
    return "shared"

context_labels["site_bias"] = context_labels.apply(site_bias, axis=1)

# Save outputs
count_by_sample.to_csv(OUT / "context_counts_by_sample.csv")
frac_by_sample.to_csv(OUT / "context_fractions_by_sample.csv")
count_by_site.to_csv(OUT / "context_counts_by_site.csv")
frac_by_site.to_csv(OUT / "context_fractions_by_site.csv")
site_enrichment.to_csv(OUT / "context_site_enrichment_ratio.csv")
top_features.to_csv(OUT / "context_top_defining_features.csv", index=False)
context_labels.to_csv(OUT / "proposed_spatial_ecological_context_labels.csv", index=False)

print("\nContext fractions by site:")
print(frac_by_site.round(3))

print("\nSite enrichment ratio:")
print(site_enrichment.round(2))

print("\nProposed context labels:")
print(context_labels[[
    "context", "proposed_label", "site_bias",
    "BM_fraction", "EM_fraction", "BM_to_EM_ratio",
    "top_positive_features"
]].to_string(index=False))

print("\nSaved outputs to:")
print(OUT)
