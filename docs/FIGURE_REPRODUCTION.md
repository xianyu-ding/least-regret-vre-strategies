# Offline figure reproduction

This route starts from the four archived ten-indicator ratings already in the repository. It does **not** call an LLM, regenerate historical prompts, or rerun TIAM, Antares, or LCA. It reconstructs figures from the supplied data, not a claim of pixel-identical published figures.

## Run

Use Python 3.10+ in a fresh virtual environment:

```sh
python -m pip install -r requirements-figures.txt
python scripts/reproduce_figures.py
python -m pytest tests/test_weighting_pipeline.py tests/test_energy_pipeline.py tests/test_figure_runner.py
```

The default command writes PNG and PDF figures to `results/figures`, validated weight tables to `results/weights`, logs to `results/logs`, and `results/figure_run_manifest.json`. It does not download anything. Input archives remain unchanged. Set `--output-dir` to a new directory to keep previous generated outputs.

A successful partial run exits 0 and explicitly records blocked figures. Unexpected exceptions, a failed child process, or a plotting script that exits without generating a figure cause exit 1. Add `--require-all` to exit 2 when any figure is blocked. This prevents “all figures succeeded” being inferred from a partial run.

## Runnable from the existing ratings

- `Fig_score_ac.py`: source-specific rating dispersion and descriptive tables.
- `Fig_weighting_details_abc.py`: hierarchical source-weight comparison.
- `Fig_weighting_sensitivity.py`: hierarchical weights and lambda sensitivity.
- `Data_weight.py`: produces the shared weight tables, not a figure itself.

All four source files are required; the runner does not replace absent sources with synthetic samples. Archived ratings are validated against the ten expected indicators and integer scores 1–5. The Zhihu archive uses nested assessment fields, while the other three use flat scores; both are supported.

## Ranking-dependent figures remain conditional

`Data_rank.py` and six downstream scripts (`Fig_ari`, `Fig_scca_sci`, `Fig_direclet_weight`, both win-rate figures, and the optional decorative comparison script) need a complete true ten-criterion scenario table. The existing summary data provide only nine criteria. A genuine CO2 series is missing, so these figures cannot yet be reproduced honestly.

Restore `data/raw/environment/co2_df.csv` with `Scenario,Pv` and all six scenario names, after confirming its unit, time period, and system boundary. Then:

```sh
python src/02_build_mcdm_dataset.py --include-co2 --output data/processed/combined_data.csv
python scripts/reproduce_figures.py
```

Alternatively pass an already verified complete table with `--indicators path/to/combined_data.csv`. Required columns are Scenario, LOLE, Cost, CO2_emission and the seven original environmental metric names. The runner never uses the separately generated nine-criterion table as a ten-criterion substitute. `--n-mc 100` can test execution after CO2 restoration, but is not a final 20,000-draw estimate.

`Fig_raking_scc_knockout.py` is explicitly blocked separately: its legacy knockout CSVs have no compatible producer in this repository. Supplying CO2 alone does not resolve that interface. It does not invent knockout tables from unrelated ARSI outputs.

## Repairs and interpretation limits

- Portable repository-relative paths replace machine-specific drive paths.
- Comma-grouped cost strings are parsed explicitly, and invalid/nonfinite/negative entries fail clearly rather than silently disappearing.
- The source-weight panel now uses the same hierarchical calculation as Data_weight. It previously normalized ten flat indicators before averaging environmental indicators, so top-level bars did not sum to one. This correction can change plotted numbers and is not merely cosmetic.
- Rank ties use average ranks and tied winners share win credit. ARSI uses a consistent convention: positive means removing the criterion worsens the alternative's rank, so that criterion helped it. These fixes may change future ranking results relative to the old code.
- Zero-weight Dirichlet components stay exactly zero; random draws use a fixed seed. Missing or malformed rating values are not filled with a score of one.
- Historical concern-tier mapping and equal-source aggregation are retained. Wilson intervals and rating dispersion are descriptive: text segments may share documents or sources, so they are not proof of independent samples or calibrated model uncertainty.
- LCIA units in the supplied tables and the LOLE reporting period must still be interpreted against the study methods. The loader does not invent unit conversions or convert intensities into totals.
- Original source images and exact manuscript plotting settings have not been supplied for pixel-level comparison. Successfully generating an image does not establish manuscript identity or validate the scientific assumptions.
