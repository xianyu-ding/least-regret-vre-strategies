# Least-Regret VRE Strategies

Research code and data accompanying the study:

> *Large-scale discourse analysis reveals least-regret integration strategies for variable renewable energy.*

This project combines LLM-assisted discourse analysis with multi-criteria decision-making (MCDM) to investigate strategies for integrating variable renewable energy (VRE). It connects priorities expressed in government, news, research and social-media texts with economic, power-system reliability and environmental indicators, and examines how strategy rankings respond to different weighting assumptions.

## Research approach

The analysis brings together three components:

1. **Discourse analysis.** Prepare and filter text from four source groups, then use an LLM to score the extent to which each text discusses specified assessment criteria.
2. **Discourse-derived weighting.** Aggregate criterion scores within and across sources to construct weights for multi-criteria analysis.
3. **Strategy evaluation and sensitivity analysis.** Combine energy-system indicators with alternative weights to examine strategy rankings, criterion contributions and robustness.

The LLM scores represent the attention given to a criterion in a text. They are not direct measurements of a strategy's technical performance. Performance indicators and discourse-derived weights enter the subsequent analysis as distinct inputs.

## Materials in this repository

- **Text preparation and filtering scripts** for government documents, news, research records and Zhihu question-and-answer data.
- **DeepSeek API scoring scripts**, including embedded prompts and output-parsing routines.
- **Filtered text and archived criterion scores** for the four source groups.
- **Weighting and visualisation code**, including source-specific comparisons and sensitivity analysis.
- **Energy-system data preparation and MCDM scripts**, alongside cost, reliability and environmental input tables.
- **An offline figure runner, its dependency list and tests**, with a guide distinguishing available figures from those that still need additional data.

### Repository layout

```text
code/
├── Text Filter/             # Text conversion, cleaning and keyword filtering
├── Deepseek Rating/         # API requests, embedded prompts and score extraction
├── Weighting/               # Earlier score-aggregation workflow
└── Figure/                  # Weight calculations, rankings and figure scripts
data/
├── raw/
│   ├── discourse/
│   │   ├── meta/            # Source files, including Git LFS-tracked assets
│   │   └── filtered/        # Filtered text by source group
│   ├── economics/           # Cost indicators
│   ├── reliability/         # Reliability indicators
│   └── environment/         # Environmental indicators
└── deepseek rating score/   # Archived scores in JSONL format
src/                        # Energy-system preparation and MCDM analysis
scripts/                    # Offline figure runner: reproduce_figures.py
tests/                      # Weighting, energy-data and figure-runner tests
docs/                       # Offline figure guide; other documents are placeholders
requirements-figures.txt    # Dependencies for the offline figure workflow
requirements.txt            # Empty general dependency-manifest placeholder
LICENSE
```

Current scoring prompts are embedded in the DeepSeek API scripts; there is no separate prompt directory. Raw TIAM inputs are not included in this repository.

## Where to start

- **Corpus preparation:** [text-filtering scripts](code/Text%20Filter) and [filtered discourse data](data/raw/discourse/filtered).
- **LLM scoring:** [DeepSeek API scripts with embedded prompts](code/Deepseek%20Rating).
- **Archived LLM results:** [criterion-score files](data/deepseek%20rating%20score).
- **Score-to-weight calculation:** [Data_weight.py](code/Figure/Data_weight.py), which groups scores into three tiers, aggregates source-level proportions and calculates hierarchical weights.
- **Weight sensitivity:** [Fig_weighting_sensitivity.py](code/Figure/Fig_weighting_sensitivity.py).
- **Energy-system analysis:** [scripts in src](src).

The repository retains scripts from different stages of the research. In particular, the archived score files contain ten indicators, whereas the currently included API prompts specify eight. These should not be treated as a fully version-matched reproduction chain.

## Using the code and data

The code is written in Python. Dependencies vary by script and include pandas, NumPy, SciPy, scikit-learn, Matplotlib, seaborn, Plotly, openpyxl, pdfplumber, requests and tqdm.

This repository is currently a research working archive, rather than a fully configured end-to-end reproduction package. Before running a script:

- Review its required inputs, output locations and dependencies.
- Check each script's path settings. Legacy scripts may require local-path changes; the offline figure runner uses repository-relative paths.
- Check that the necessary source data are available. Some energy-system inputs are not included, and some source assets are tracked through Git LFS.
- For API-based scoring, use your own credentials securely and review the provider's current access requirements and costs.

The general `requirements.txt` file is empty, and several methodology and workflow documents remain placeholders. For the offline figure workflow, use the populated [requirements-figures.txt](requirements-figures.txt) and [figure-reproduction guide](docs/FIGURE_REPRODUCTION.md). This dependency list does not configure every legacy script or the full upstream workflow. Archived results can be inspected without making new API requests.

## Interpretation and reproducibility

The weighting and ranking scripts support exploration of methodological choices and sensitivity. These analyses should be distinguished from validation of the LLM's document-level scores.

The repository does not currently provide a complete record of corpus collection, prompt development, manual annotation or repeated model runs. Users should account for these limits when interpreting results or adapting the approach to another corpus.

## License and data use

The source code is released under the [MIT License](LICENSE).

Source documents and third-party datasets may be subject to separate copyright, access or redistribution conditions. Their inclusion in this repository does not establish a blanket licence for reuse; check the relevant source terms before redistributing them.

## Citation

The study is identified in this repository as:

> Ding, Q. et al. *Large-scale discourse analysis reveals least-regret integration strategies for variable renewable energy.*

Journal and DOI details are not currently provided in this repository.


## Reproducing figures offline

For commands, available figures, and remaining data requirements, see [Offline figure reproduction](docs/FIGURE_REPRODUCTION.md). This route uses the archived scores without model calls or upstream simulations.

The source-score and weighting figures have an archived-data workflow. Ranking-dependent figures still require genuine CO2 data to complete the indicator table, and the legacy knockout figure additionally needs compatible knockout tables. The runner reports blocked figures explicitly; a partial run is not a complete reproduction of all figures or a claim of pixel-identical manuscript figures.
