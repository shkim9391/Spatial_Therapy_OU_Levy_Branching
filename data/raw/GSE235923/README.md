# GSE235923 – Pediatric AML Longitudinal Single-Cell RNA-seq

## Overview

This directory contains the processed files derived from **GEO accession GSE235923**, which are used for **therapy-response validation and branching-associated ecological analyses** in this repository.

**GEO Accession:** GSE235923

**Title:**
*Single-cell analysis reveals altered tumor microenvironments of relapse- and remission-associated pediatric acute myeloid leukemia*

**Publication**

Mumme H, Thomas BE, Bhasin SS, Krishnan U, et al.

*Single-cell analysis reveals altered tumor microenvironments of relapse- and remission-associated pediatric acute myeloid leukemia.*

**Nature Communications** 2023;14:6209.

PMID: 37798266

---

## Original study

The original study investigated pediatric AML using longitudinal single-cell RNA sequencing collected at:

- Diagnosis (Dx)
- End of Induction (EOI)
- Relapse

The study characterized leukemia-associated transcriptional programs, immune microenvironment remodeling, and therapy-associated cellular states linked to relapse and continuous complete remission (CCR). GSE235923 GEO Accession viewer.pdf

---

## Experimental design

Bone marrow samples were profiled using

- 10x Genomics Single Cell 3′ RNA sequencing

The GEO series includes **31 single-cell samples** collected across diagnosis, end-of-induction, and relapse time points. GSE235923 GEO Accession viewer.pdf

Because of patient privacy, **raw sequencing reads are not distributed through GEO**. Processed count matrices and annotations are provided instead. GSE235923 GEO Accession viewer.pdf

---

## Files available from GEO

The GEO submission provides

- processed expression matrices
- MTX/TSV files
- sample annotations
- processed single-cell datasets

Primary download:

- `GSE235923_RAW.tar`

---

## Use in this repository

Unlike GSE279576, this dataset is **not used for ecological context discovery**.

Instead, it is used as an **independent biological validation dataset** for evaluating ecological hypotheses derived from spatial transcriptomics.

Within this repository, GSE235923 is used to

1. project leukemia cells onto the ecological reference learned from GSE279576;
2. quantify ecological context occupancy across therapy-associated samples;
3. evaluate branching-like ecological amplification;
4. compare diagnosis, treatment, and relapse-associated ecological states;
5. summarize therapy-associated ecological changes.

The processed derivative tables generated from this dataset include

- therapy response metadata
- projected ecological context assignments
- branching summaries
- therapy-response summary tables

---

## Processing pipeline

Typical scripts include

```
scripts/
    build_Figure6_branching_amplification.py
    build_Figure7_therapy_response_validation.py
```

Outputs include

- ecological context projections
- branching summaries
- therapy-response comparisons
- publication figures
- supplementary data tables

---

## Data source

NCBI Gene Expression Omnibus (GEO)

Accession:

GSE235923

---

## Citation

Please cite the original publication if you use this dataset:

> Mumme H, Thomas BE, Bhasin SS, Krishnan U, et al.
> Single-cell analysis reveals altered tumor microenvironments of relapse- and remission-associated pediatric acute myeloid leukemia.
> Nature Communications. 2023;14:6209.

---

## Notes

This repository performs **secondary computational analyses** using publicly available processed single-cell data.

Ecological context labels used in this project are generated independently and are **not part of the original GEO submission**.
