# k-scale: k-Anonymizing Millions of Trajectories

This repository contains the implementation, evaluation pipeline, and benchmarking notebooks for the paper:

**“k-scale: k-Anonymizing Millions of Trajectories”**  
by Abhishek Kumar Mishra and Marco Fiore, published at the IEEE INFOCOM 2026.

The work introduces **k-scale**, a scalable framework for efficiently achieving trajectory k-anonymity on datasets containing millions of mobility traces while preserving strong analytical utility.

---

# Paper

- HAL Archive: https://hal.science/hal-05407897
- PDF: https://hal.science/hal-05407897v1/file/Efficiently_k_anonymizing_a_Million_Trajectories___IEEE_INFOCOM__HAL_Version_.pdf

If you use this repository, please cite the paper below.

---

# Repository Structure

```text
.
├── Benchmarks/
├── DatasetGIS/
├── Evaluation/
├── Results/
├── Source/
└── README.md
```

---

# Folder Description

## `Source/`

Contains the core implementation of the **k-scale** framework. 

---

## `Benchmarks/`

Contains notebooks and scripts used to compare **k-scale** against state-of-the-art trajectory anonymization baselines.

---

## `DatasetGIS/`

Contains example notebooks demonstrating how to preprocess raw mobility datasets (e.g., CDR datasets). 

---

## `Evaluation/`

Contains the notebooks used to reproduce the experimental results reported in the paper. 

---

## `Results/`

Contains the generated experimental outputs. 

---

# Reproducing the Paper Results

The general workflow is:

```text
Raw Dataset
    ↓
DatasetGIS preprocessing
    ↓
k-scale anonymization (Source/)
    ↓
Evaluation notebooks
    ↓
Results generation
```

For demonstration, we have used the public mobility dataset: YJMob100K ({https://www.nature.com/articles/s41597-024-03237-9, https://zenodo.org/records/10142719}) in the source code denoted by the variable taka_dataset, but a user can replace the loading of this variable to any source dataset and the anonymuzation pipeline will work. 

---

# Requirements

The implementation is primarily based on Python notebooks and scientific Python libraries.

Typical dependencies include:

```bash
numpy
pandas
scikit-learn
geopandas
matplotlib
scipy
jupyter
```

Depending on the evaluation notebook, additional packages may be required.

---

# Citation

Please cite the following paper if you use this repository or build upon this work:

```bibtex
@inproceedings{mishra:hal-05407897,
  title     = {k-scale: k-Anonymizing Millions of Trajectories},
  author    = {Mishra, Abhishek Kumar and Fiore, Marco},
  booktitle = {IEEE INFOCOM 2026 Conference Proceedings},
  address   = {Tokyo, Japan},
  year      = {2026},
  month     = may,
  url       = {https://hal.science/hal-05407897},
  pdf       = {https://hal.science/hal-05407897v1/file/Efficiently_k_anonymizing_a_Million_Trajectories___IEEE_INFOCOM__HAL_Version_.pdf},
  hal_id    = {hal-05407897},
  hal_version = {v1}
}
```

---

# Acknowledgment

This repository accompanies the research work presented at IEEE INFOCOM 2026 and is intended to support reproducible research in large-scale mobility privacy and trajectory anonymization. 