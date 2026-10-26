# Engine RUL Intelligence

An educational predictive-maintenance portfolio project that estimates remaining useful life (RUL), measured in operating cycles, for turbofan engines using the NASA C-MAPSS FD001–FD004 benchmark and a pooled XGBoost regression model.

> **Scope and safety:** This is a research/portfolio prototype using simulated benchmark trajectories. It is not a certified aviation-maintenance tool, does not provide maintenance instructions, and must not be used to make real aircraft safety decisions.

## Dashboard

The Streamlit dashboard provides:

- Fleet overview and a maintenance-priority queue based on configurable demonstration thresholds.
- Filters for the four C-MAPSS subsets: FD001, FD002, FD003, and FD004.
- Engine-level RUL history and selected sensor readings.
- Predicted-RUL distribution and predicted-versus-benchmark visualizations for retrospective analysis.
- Model facts and XGBoost feature-importance summaries.

Run the dashboard from the project root:

```powershell
streamlit run app.py
```

## Current model configuration

The current saved artifact was trained in Notebook 10 and is stored at:

```text
models/xgb_rul_model_all_subsets.joblib
```

| Setting | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Data | NASA C-MAPSS FD001–FD004 training trajectories |
| Training engines | 709 |
| Training rows | 160,359 |
| Feature count | 140 |
| Feature strategy | Regime-aware features |
| Rolling window | 10 operating cycles |
| Target cap | 250 cycles |
| Objective | `reg:pseudohubererror` |
| Huber slope | 10 |
| Sample weighting | Row-uniform |

The feature pipeline uses chronological, engine-grouped rolling statistics. Target values are used for supervised training only and are excluded from the model input features. Regime assignment and related transformation statistics are fitted on training trajectories and saved with the deployment artifact so inference can use the same transformation.

## Project workflow

The notebooks record the progression from data understanding and baseline model development through test evaluation, robustness checks, deployment-matched validation, loss-function experiments, target-cap sensitivity, NASA-style score auditing, repeated engine-level validation, and final model fitting.

Typical workflow:

1. Inspect and audit the C-MAPSS files.
2. Build and evaluate the baseline pooled XGBoost model.
3. Run robustness and engine-level validation experiments.
4. Audit the NASA-style scoring formula and aggregation.
5. Compare target caps using development/validation splits.
6. Fit and save the selected cap-250 model using all labelled training trajectories.
7. Launch the dashboard and verify all four subsets.

## Data placement

Place the benchmark text files in the project-root `CMAPSSData/` directory. The notebooks expect the standard training and test files for FD001–FD004. Do not commit the raw dataset unless its distribution terms permit it; document where the dataset was obtained and how to place it locally.

## Validation and evaluation notes

Target-cap experiments compared `cap_250`, `cap_350`, and an uncapped target across three repeated engine-level validation splits while excluding the locked audit engines from each repetition. Across those splits, `cap_250` had the best mean macro-subset MAE (about 27.08 cycles) and mean macro-subset RMSE (about 31.39 cycles), while `cap_350` had the lowest average macro NASA-style score of the three candidates. This is a metric trade-off, not a universal winner; cap 250 was selected provisionally with macro MAE as the primary metric.

The dashboard currently displays a retrospective official-test MAE of about 21.5 cycles for the loaded model. Official test labels have already been inspected during development, so treat this as a reported retrospective benchmark result—not as a pristine blind estimate and not as a basis for further tuning. The dashboard labels benchmark truth as offline-only and includes a warning for the saved Notebook 03 metrics because those metrics came from a different model configuration (target cap 200).

The test-engine benchmark RUL shown in the dashboard is available only for offline retrospective analysis. At real prediction time, the true RUL is not known.

## Run locally

1. Create/activate the project's Python environment.
2. Install the dependencies listed in `requirements.txt`.
3. Confirm `CMAPSSData/` contains the expected benchmark files and `models/xgb_rul_model_all_subsets.joblib` exists.
4. Launch:

   ```powershell
   streamlit run app.py
   ```

5. Select FD001, FD002, FD003, and FD004 to view all 707 official test engines.

## Repository hygiene before publishing

- Keep virtual environments (`renv/`, `.venv/`) out of Git.
- Keep Python caches, notebook checkpoints, temporary exports, and local logs out of Git.
- Check the dataset's redistribution terms before committing raw C-MAPSS files.
- Decide deliberately whether to publish the trained model artifact; if included, document its origin and configuration.
- Do not publish machine-specific absolute paths or credentials.
- Include a dashboard screenshot in the repository README if appropriate.

## Known limitations

- Benchmark-derived data is simulated; real-world distribution shift is not assessed.
- RUL errors vary across operating regimes/subsets and individual engines.
- Feature importance is a model diagnostic, not causal evidence.
- Configurable policy thresholds are demonstration controls, not learned failure probabilities or approved maintenance policy.
- A low aggregate MAE does not guarantee a safe prediction for every engine.
