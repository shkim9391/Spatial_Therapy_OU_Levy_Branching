# GSE253355 – Normal Human Bone Marrow Reference Atlas

## Overview

This directory contains the raw and processed files derived from **GEO accession GSE253355**, which serves as the **normal bone marrow ecological reference** for the spatial ecological modeling framework developed in this repository.

**GEO Accession:** GSE253355

**Title:**
*Mapping the Cellular Biogeography of Human Bone Marrow Niches Using Single-Cell Transcriptomics*

**Publication:**

Bandyopadhyay S, Duffy MP, Ahn KJ, Sussman JH, et al.
*Mapping the cellular biogeography of human bone marrow niches using single-cell transcriptomics and proteomic imaging.*
**Cell** 2024;187(12):3120–3140.e29.

PMID: 38714197

---

## Original study

The original study generated a comprehensive single-cell and spatial atlas of healthy human bone marrow by integrating:

- Single-cell RNA sequencing (scRNA-seq)
- CODEX multiplex proteomic imaging
- Spatial neighborhood analysis
- Cell–cell interaction inference

The atlas characterizes both hematopoietic and non-hematopoietic compartments and provides a high-resolution reference of normal bone marrow organization. GSE253355 GEO Accession viewer.pdf

---

## Experimental design

Samples were obtained from **hematologically healthy individuals undergoing total hip arthroplasty**.

The study profiled:

- enriched mesenchymal cells
- hematopoietic stem and progenitor cells (HSPCs)
- RBC-depleted bone marrow cells

using

- 10x Genomics Single Cell 3′ v3.1
- Illumina NovaSeq 6000 sequencing

The published dataset includes **12 single-cell samples** together with processed Seurat objects and raw count matrices. GSE253355 GEO Accession viewer.pdf

---

## Files available from GEO

The GEO submission provides:

- Raw 10x Genomics count matrices
- Processed Seurat objects
- Cell annotations
- Gene-expression matrices

Major supplementary files include:

- `GSE253355_RAW.tar`
- `GSE253355_Normal_Bone_Marrow_Atlas_Seurat_SB_v2.rds.gz`
- `GSE253355_MSC_Subset_Seurat.rds.gz`

These files can be downloaded directly from the GEO accession page. GSE253355 GEO Accession viewer.pdf

---

## Use in this repository

This project uses **GSE253355 exclusively as a normal bone marrow ecological reference**.

The dataset is **not analyzed for biological discovery itself**, but is used to derive ecological gene-program signatures that serve as references for downstream spatial leukemia analyses.

Specifically, this dataset is used to:

1. perform quality control and preprocessing;
2. identify major normal bone marrow cellular programs;
3. construct ecological reference signatures;
4. compute ecological program scores for leukemia spatial transcriptomic samples (GSE279576).

No disease samples from GSE253355 are included because the dataset consists of healthy human bone marrow.

---

## Processing pipeline

Within this repository, GSE253355 is processed using scripts similar to:

```
scripts/
    process_GSE253355_raw.py
    score_GSE253355_normal_BM_ecology.py
```

Outputs include:

- processed AnnData objects
- ecological gene-program matrices
- reference ecological signatures
- quality-control summaries

These processed outputs are subsequently used throughout the spatial ecological analysis pipeline.

---

## Data source

NCBI Gene Expression Omnibus (GEO)

Accession:

GSE253355

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE253355

---

## Citation

If you use this dataset, please cite the original publication:

> Bandyopadhyay S, Duffy MP, Ahn KJ, Sussman JH, et al.
> Mapping the cellular biogeography of human bone marrow niches using single-cell transcriptomics and proteomic imaging.
> Cell. 2024;187(12):3120–3140.e29.

---

## Notes

This repository does **not** claim ownership of the original data.

The raw sequencing data and associated metadata remain available through the NCBI Gene Expression Omnibus (GEO). This project performs **secondary computational analyses** using publicly available datasets in accordance with GEO data-sharing policies.
