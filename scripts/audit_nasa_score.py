"""Audit the validation NASA-style asymmetric RUL score without retraining a model.

Expected inputs (created by notebooks/07_target_cap_sensitivity.ipynb):
  reports/target_cap_validation_predictions.csv
  reports/target_cap_validation_comparison.csv

Outputs:
  reports/nasa_score_audit_summary.csv
  reports/nasa_score_audit_by_subset.csv
  reports/nasa_score_audit_top_errors.csv

The audit re-computes the penalty from true/predicted RUL, verifies the saved
per-row penalty, checks aggregation levels, and identifies high-impact cases.
It never reads official test RUL labels and never writes/overwrites a model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREDICTIONS_FILE = "target_cap_validation_predictions.csv"
COMPARISON_FILE = "target_cap_validation_comparison.csv"
REQUIRED_COLUMNS = {
    "candidate_id", "subset", "engine_key", "unit_nr", "cutoff_cycle",
    "true_rul_cycles", "predicted_rul_cycles", "nasa_penalty",
}


def nasa_penalty(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NASA C-MAPSS-style asymmetric penalty and exponent.

    Let d = prediction - truth. Under-prediction (d < 0) uses scale 13;
    over-prediction (d >= 0) uses scale 10. Exponent clipping at 50 is a
    numerical safeguard against overflow and should not affect ordinary cases.
    """
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    penalty = np.expm1(np.clip(exponent, 0.0, 50.0))
    return penalty, exponent


def _find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "reports").is_dir():
            return candidate
    # Common invocation: running this script from scripts/.
    if here.name == "scripts" and (here.parent / "reports").is_dir():
        return here.parent
    return here


def run_audit(reports_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the score audit and return output DataFrames and written paths."""
    reports = Path(reports_dir).resolve() if reports_dir else (_find_project_root() / "reports")
    predictions_path = reports / PREDICTIONS_FILE
    comparison_path = reports / COMPARISON_FILE
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Could not find {predictions_path}. Run notebooks/07_target_cap_sensitivity.ipynb first."
        )
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Could not find {comparison_path}. Run notebooks/07_target_cap_sensitivity.ipynb first."
        )

    predictions = pd.read_csv(predictions_path)
    missing = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing)}")
    comparison = pd.read_csv(comparison_path)
    if "candidate_id" not in comparison.columns:
        raise ValueError("Candidate summary CSV has no 'candidate_id' column.")

    numeric_columns = ["true_rul_cycles", "predicted_rul_cycles", "nasa_penalty"]
    for column in numeric_columns:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    if predictions[numeric_columns].isna().any().any():
        bad_count = int(predictions[numeric_columns].isna().any(axis=1).sum())
        raise ValueError(f"Found {bad_count} rows with missing/non-numeric RUL or penalty values.")
    if not np.isfinite(predictions[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Found non-finite RUL or saved penalty values in prediction CSV.")

    y_true = predictions["true_rul_cycles"].to_numpy(dtype=float)
    y_pred = predictions["predicted_rul_cycles"].to_numpy(dtype=float)
    saved_penalty = predictions["nasa_penalty"].to_numpy(dtype=float)
    recomputed_penalty, exponent = nasa_penalty(y_true, y_pred)
    predictions["audit_error_pred_minus_true"] = y_pred - y_true
    predictions["audit_abs_error"] = np.abs(y_pred - y_true)
    predictions["audit_exponent"] = exponent
    predictions["audit_nasa_penalty"] = recomputed_penalty
    predictions["audit_penalty_difference"] = saved_penalty - recomputed_penalty

    # Tiny values are expected to agree; this detects formula/sign/column drift.
    penalty_abs_diff = np.abs(saved_penalty - recomputed_penalty)
    formula_match = bool(np.allclose(saved_penalty, recomputed_penalty, rtol=1e-10, atol=1e-8))

    # Mirror Notebook 07 aggregation: average cutoffs within each engine, then
    # engines equally within each subset, then take an unweighted subset mean.
    engine = predictions.groupby(["candidate_id", "subset", "engine_key"], as_index=False).agg(
        engine_cases=("audit_nasa_penalty", "size"),
        engine_mean_nasa_penalty=("audit_nasa_penalty", "mean"),
    )
    by_subset = engine.groupby(["candidate_id", "subset"], as_index=False).agg(
        engines=("engine_key", "nunique"),
        cases=("engine_cases", "sum"),
        pooled_nasa_mean=("engine_mean_nasa_penalty", "mean"),
    )
    macro = by_subset.groupby("candidate_id", as_index=False).agg(
        recomputed_macro_subset_nasa_score_per_case=("pooled_nasa_mean", "mean"),
        subsets_scored=("subset", "nunique"),
    )
    pooled = predictions.groupby("candidate_id", as_index=False).agg(
        n_cases=("audit_nasa_penalty", "size"),
        pooled_row_mean_nasa_penalty=("audit_nasa_penalty", "mean"),
        pooled_row_sum_nasa_penalty=("audit_nasa_penalty", "sum"),
        max_abs_error_cycles=("audit_abs_error", "max"),
        max_single_case_nasa_penalty=("audit_nasa_penalty", "max"),
        max_exponent=("audit_exponent", "max"),
        cases_at_exponent_cap=("audit_exponent", lambda s: int(np.sum(np.asarray(s) >= 50.0))),
        true_rul_max=("true_rul_cycles", "max"),
        predicted_rul_max=("predicted_rul_cycles", "max"),
    )

    # How much of the total unweighted row penalty is explained by the worst cases?
    concentration_rows = []
    top_errors = []
    keep_cols = [
        "candidate_id", "subset", "engine_key", "unit_nr", "cutoff_cycle",
        "true_rul_cycles", "predicted_rul_cycles", "audit_error_pred_minus_true",
        "audit_abs_error", "audit_exponent", "audit_nasa_penalty",
    ]
    for candidate_id, frame in predictions.groupby("candidate_id", sort=False):
        ordered = frame.sort_values("audit_nasa_penalty", ascending=False).copy()
        total = float(ordered["audit_nasa_penalty"].sum())
        row = {"candidate_id": candidate_id}
        for k in (1, 5, 10):
            top_sum = float(ordered["audit_nasa_penalty"].head(k).sum())
            row[f"top_{k}_penalty_share_pct"] = (100.0 * top_sum / total) if total > 0 else 0.0
        concentration_rows.append(row)
        ordered["share_of_candidate_total_penalty_pct"] = (
            100.0 * ordered["audit_nasa_penalty"] / total if total > 0 else 0.0
        )
        top_errors.append(ordered[keep_cols + ["share_of_candidate_total_penalty_pct"]].head(20))
    concentration = pd.DataFrame(concentration_rows)

    summary = pooled.merge(macro, on="candidate_id", how="left").merge(concentration, on="candidate_id", how="left")
    if "macro_subset_nasa_score_per_case" in comparison.columns:
        summary = summary.merge(
            comparison[["candidate_id", "macro_subset_nasa_score_per_case"]],
            on="candidate_id", how="left", suffixes=("", "_from_notebook"),
        )
        summary["macro_score_difference_vs_notebook"] = (
            summary["recomputed_macro_subset_nasa_score_per_case"]
            - summary["macro_subset_nasa_score_per_case"]
        )
    else:
        summary["macro_subset_nasa_score_per_case"] = np.nan
        summary["macro_score_difference_vs_notebook"] = np.nan

    # Add main predictive metrics for easy comparison in the same CSV.
    metric_columns = [
        c for c in ["candidate_id", "pooled_mae", "pooled_rmse", "pooled_r2",
                    "macro_subset_mae", "macro_subset_rmse", "macro_subset_late_pct",
                    "macro_subset_mean_bias"] if c in comparison.columns
    ]
    if len(metric_columns) > 1:
        summary = summary.merge(comparison[metric_columns], on="candidate_id", how="left")

    by_subset_out = by_subset.merge(
        predictions.groupby(["candidate_id", "subset"], as_index=False).agg(
            row_level_nasa_mean=("audit_nasa_penalty", "mean"),
            max_abs_error_cycles=("audit_abs_error", "max"),
            max_single_case_nasa_penalty=("audit_nasa_penalty", "max"),
        ), on=["candidate_id", "subset"], how="left"
    )
    top_errors_out = pd.concat(top_errors, ignore_index=True) if top_errors else pd.DataFrame()

    summary_path = reports / "nasa_score_audit_summary.csv"
    subset_path = reports / "nasa_score_audit_by_subset.csv"
    top_path = reports / "nasa_score_audit_top_errors.csv"
    summary.to_csv(summary_path, index=False)
    by_subset_out.to_csv(subset_path, index=False)
    top_errors_out.to_csv(top_path, index=False)

    print("NASA-style validation score audit")
    print("=" * 36)
    print(f"Prediction rows audited: {len(predictions):,}")
    print(f"Candidates: {predictions['candidate_id'].nunique()}")
    print(f"Per-row saved penalty matches formula: {'YES' if formula_match else 'NO'}")
    print(f"Maximum absolute saved-vs-recomputed penalty difference: {penalty_abs_diff.max():.6g}")
    if formula_match:
        print("Formula/sign check passed: d = prediction - truth; under-prediction /13, over-prediction /10.")
    else:
        print("WARNING: saved penalty differs from the standard formula. Inspect audit_penalty_difference and source columns.")
    if "macro_score_difference_vs_notebook" in summary:
        delta = summary["macro_score_difference_vs_notebook"].abs().max()
        print(f"Maximum macro aggregation difference vs Notebook 07: {delta:.6g}")
        if pd.notna(delta) and delta > max(1e-6, 1e-8 * max(1.0, float(summary['recomputed_macro_subset_nasa_score_per_case'].max()))):
            print("WARNING: aggregation does not match Notebook 07; inspect candidate rows and grouping keys.")
        else:
            print("Macro aggregation check passed.")
    print("\nCandidate diagnostics (selected columns):")
    view_cols = [
        "candidate_id", "pooled_mae", "pooled_rmse", "recomputed_macro_subset_nasa_score_per_case",
        "pooled_row_mean_nasa_penalty", "pooled_row_sum_nasa_penalty",
        "max_abs_error_cycles", "max_single_case_nasa_penalty",
        "top_1_penalty_share_pct", "top_5_penalty_share_pct", "cases_at_exponent_cap",
    ]
    view_cols = [c for c in view_cols if c in summary.columns]
    print(summary[view_cols].sort_values("recomputed_macro_subset_nasa_score_per_case").to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\nFiles written:")
    print(f" - {summary_path}")
    print(f" - {subset_path}")
    print(f" - {top_path}")
    print("\nInterpretation: the validation score is averaged per engine and subset; it is not the official benchmark's raw total score.")
    print("No model was retrained, changed, or saved by this audit.")

    return {
        "candidate_summary": summary,
        "by_subset": by_subset_out,
        "top_errors": top_errors_out,
        "formula_matches": formula_match,
        "max_penalty_difference": float(penalty_abs_diff.max()),
        "output_paths": [summary_path, subset_path, top_path],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir", type=Path, default=None,
        help="Directory containing the Notebook 07 CSV outputs (default: ./reports).",
    )
    args = parser.parse_args()
    run_audit(args.reports_dir)


if __name__ == "__main__":
    main()
