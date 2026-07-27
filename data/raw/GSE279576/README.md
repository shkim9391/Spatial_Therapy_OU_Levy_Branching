# GSE279576 – Spatial Transcriptomics of Acute Myeloid Leukemia

## Overview

This directory contains the raw and processed files derived from **GEO accession GSE279576**, which serves as the **primary spatial transcriptomic dataset** for the therapy-aware spatial ecological modeling framework developed in this repository.

**GEO Accession:** GSE279576

**Title:**
*Integrative Spatial Multi-Omics Reveal Niche-Specific Inflammatory Signaling and Differentiation Hierarchies in Acute Myeloid Leukemia*

**Publication**

Dasdemir E, Veletic I, Ly CP, Quesada AE, et al.

*Integrative spatial multi-omics reveal niche-specific inflammatory signaling and differentiation hierarchies in AML.*

**iScience** 2026;29(1):114289.

PMID: 41536977

---

## Original study

The original study generated a multimodal spatial atlas of acute myeloid leukemia (AML) using:

- 10x Genomics Visium Spatial Transcriptomics
- GeoMX Digital Spatial Profiling
- Opal multiplex immunofluorescence
- Cell–cell communication analysis

The study compared **bone marrow (BM)** and **extramedullary (EM)** AML tissues to investigate inflammatory signaling, differentiation hierarchies, and spatial niche organization. GSE279576 GEO Accession viewer.pdf

---

## Experimental design

Bone marrow and extramedullary tissue biopsies were collected from **two newly diagnosed AML patients before treatment**.

Spatial transcriptomic profiling was performed using:

- Visium Spatial Gene Expression v1
- Visium CytAssist Spatial Gene Expression v2
- Illumina NovaSeq 6000 sequencing

The GEO series contains **six Visium samples** representing BM and EM tissues. GSE279576 GEO Accession viewer.pdf

---

## Files available from GEO

The GEO submission provides:

- Raw Visium output files
- Spatial count matrices
- HDF5 files
- Tissue images
- Spatial metadata

Primary download:

- `GSE279576_RAW.tar`

---

## Use in this repository

**GSE279576 is the central dataset used throughout this repository.**

It is used to construct and evaluate the complete spatial ecological modeling framework, including:

1. ecological program scoring
2. spatial ecological context discovery
3. ecological feature profiling
4. evolutionary opportunity scoring
5. OU-like ecological retention analysis
6. Lévy-like ecological escape analysis
7. branching-like ecological amplification
8. therapy-associated ecological interpretation

The analyses are interpreted as **cross-sectional ecological summaries rather than direct temporal measurements of tumor evolution**.

---

## Processing pipeline

Typical processing scripts include

```
scripts/
    process_GSE279576_visium.py
    discover_GSE279576_spatial_ecological_contexts.py
    build_Figure3_opportunity.py
    build_Figure4_ou_like_spatial_summary.py
    build_Figure5_levy_like_jumps.py
    build_Figure6_branching_amplification.py
```

Outputs include

- processed AnnData objects
- ecological program scores
- ecological context labels
- neighborhood graphs
- opportunity scores
- OU-like summaries
- Lévy-like escape summaries
- branching-like amplification statistics
- publication-quality figures

---

## Data source

NCBI Gene Expression Omnibus (GEO)

Accession:

GSE279576

---

## Citation

Please cite the original publication if you use this dataset:

> Dasdemir E, Veletic I, Ly CP, Quesada AE, et al.
> Integrative spatial multi-omics reveal niche-specific inflammatory signaling and differentiation hierarchies in AML.
> iScience. 2026;29(1):114289.

---

## Notes

This repository performs **secondary computational analyses** using publicly available spatial transcriptomic data.

The original data remain available through NCBI GEO.
