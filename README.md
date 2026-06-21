# Least-Regret VRE Strategies

This repository contains the code, data and documentation supporting the study:

> *Large-scale discourse analysis reveals least-regret integration strategies for variable renewable energy.*

The project combines large-scale discourse analysis with multi-criteria decision-making (MCDM) to examine robust, least-regret strategies for integrating variable renewable energy (VRE).

## Overview

The workflow integrates:

1. Preparation of energy-system model outputs.
2. Construction of a combined dataset covering economic, reliability and environmental indicators.
3. Large-scale discourse analysis of VRE integration strategies.
4. LLM-assisted classification and coding of discourse evidence.
5. Stochastic competitive contribution (SCC) and MCDM analysis.
6. Robustness analysis and generation of manuscript figures and tables.

## Repository structure

```text
least-regret-vre-strategies/
├── data/
│   ├── raw/                  # Raw input data where redistribution is permitted
│   ├── processed/            # Cleaned and analysis-ready datasets
│   ├── demo/                 # Small example data for testing the workflow
│   └── README_data.md        # Data documentation and source information
├── src/
│   ├── 01_prepare_tiam_outputs.py
│   ├── 02_build_mcdm_dataset.py
│   ├── 03_run_scc_analysis.py
│   └── 04_visualise_normalisation.py
├── prompts/                  # LLM prompts and coding instructions
├── docs/                     # Methodological and reproducibility documentation
├── results/
│   ├── tables/
│   ├── figures/
│   └── supplementary_figures/
├── notebooks/                # Optional exploratory notebooks
├── requirements.txt
├── LICENSE
└── README.md
```

## Data availability

The repository includes processed datasets, analysis-ready CSV/Excel files, output tables and documentation required to reproduce the main analyses.

Some original source documents or third-party records may not be redistributed because of copyright, database-access or licensing restrictions. For such sources, this repository provides metadata, screening records, processing scripts and documentation to enable reconstruction of the analytical dataset from the sources described in the manuscript.

Detailed information on each dataset is provided in `data/README_data.md`.

## Software requirements

The analysis was implemented in Python.

Key packages include:

* pandas
* numpy
* scipy
* scikit-learn
* matplotlib
* seaborn
* plotly
* openpyxl

A complete list of dependencies will be provided in `requirements.txt`.

The workflow is designed to run on a standard personal computer. No specialised hardware is required.

## Installation

Clone the repository:

```bash
git clone https://github.com/xianyu-ding/least-regret-vre-strategies.git
cd least-regret-vre-strategies
```

Create and activate a virtual environment:

```bash
python -m venv vre_env
source vre_env/bin/activate
```

For Windows:

```bash
vre_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical installation time should be less than 10 minutes on a standard personal computer.

## Reproducing the analysis

The main workflow should be run in the following sequence:

```bash
python src/01_prepare_tiam_outputs.py
python src/02_build_mcdm_dataset.py
python src/03_run_scc_analysis.py
python src/04_visualise_normalisation.py
```

The scripts perform the following tasks:

1. Prepare TIAM-derived capacity and generation outputs.
2. Combine cost, reliability, emissions and environmental-impact indicators into the MCDM input dataset.
3. Run SCC/MCDM analysis, stochastic weight sampling, robustness checks and figure generation.
4. Produce visualisations illustrating the normalisation process used in the MCDM workflow.

Outputs are written to:

```text
results/tables/
results/figures/
results/supplementary_figures/
```

## LLM-assisted discourse analysis

The discourse-analysis materials will be added to this repository, including:

* corpus metadata and screening records;
* document selection and filtering criteria;
* prompts and coding instructions;
* model settings and post-processing procedures;
* classified discourse outputs and validation materials.

These materials will be organised in:

```text
data/raw/discourse/
data/processed/
prompts/
docs/llm_workflow.md
docs/data_collection_and_screening.md
```

## MCDM and SCC framework

The MCDM framework evaluates VRE integration strategies using economic, technical and environmental criteria.

The analysis includes stochastic weight sampling to assess how scenario rankings change under uncertainty in decision-maker preferences. The SCC approach evaluates the importance of criteria by measuring how often removing a criterion changes the selected best-performing strategy.

Further methodological details will be provided in:

```text
docs/methodology.md
docs/variable_definitions.md
```

## Reproducibility

To reproduce the main analysis, install the required Python dependencies, place the required input files in the designated data directories, and run the scripts in numerical order.

The repository will include processed data, scripts, intermediate outputs, final tables and figure-generation workflows wherever redistribution is permitted.

## License

The source code is released under the MIT License.

Data are provided for research and reproducibility purposes. Please consult `data/README_data.md` for any restrictions relating to third-party data or source documents.

## Citation

If you use this repository, please cite:

> Ding, Q. et al. *Large-scale discourse analysis reveals least-regret integration strategies for variable renewable energy.* [Journal details and DOI to be added after publication].

## Contact

For questions concerning the code, data or reproducibility workflow, please contact the corresponding author through the publication record.
