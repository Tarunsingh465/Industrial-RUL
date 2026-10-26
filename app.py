"""Interactive dashboard for pooled XGBoost RUL prediction across NASA C-MAPSS FD001-FD004.

Run from the project root after executing notebooks 01, 02, 04 and then 03:
    streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Engine RUL Intelligence", page_icon="🛠️", layout="wide")
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
CMAPSS_DIR = ROOT / "CMAPSSData"
MODEL_PATH = ROOT / "models" / "xgb_rul_model_all_subsets.joblib"
REPORTS_DIR = ROOT / "reports"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
COLUMN_NAMES = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + [f"s_{i}" for i in range(1, 22)]


def locate_file(filename):
    for path in (RAW_DIR / filename, CMAPSS_DIR / filename, ROOT / filename):
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {filename}. Run Notebook 01 or copy all C-MAPSS .txt files to data/raw/.")


def load_table(filename):
    path = locate_file(filename)
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES, engine="python").dropna(axis=1, how="all")


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("The pooled model is missing. Run notebooks/02_xgboost_rul_training.ipynb and notebooks/04_robustness_experiments.ipynb first.")
    return joblib.load(MODEL_PATH)


def engineer_features(frame, sensor_cols, setting_cols, window, subsets):
    ordered = frame.sort_values(["subset", "unit_nr", "time_cycles"]).reset_index(drop=True).copy()
    features = ordered[["subset", "unit_nr", "engine_key", "time_cycles"] + list(setting_cols) + list(sensor_cols)].copy()
    groups = ordered.groupby(["subset", "unit_nr"], sort=False)
    for sensor in sensor_cols:
        signal = groups[sensor]
        features[f"{sensor}_roll_mean_{window}"] = signal.transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        features[f"{sensor}_roll_std_{window}"] = signal.transform(lambda s: s.rolling(window=window, min_periods=2).std(ddof=0)).fillna(0.0)
        features[f"{sensor}_delta_1"] = signal.diff().fillna(0.0)
    for subset in subsets:
        features[f"subset_{subset}"] = (ordered["subset"] == subset).astype(np.int8)
    return features



def add_regime_features_for_artifact(features, transformer):
    """Apply the regime transformer learned exclusively on training engines."""
    out = features.copy().reset_index(drop=True)
    scaled = transformer["setting_scaler"].transform(out[["setting_1", "setting_2", "setting_3"]])
    regime = transformer["kmeans"].predict(scaled).astype(int)
    for r in range(int(transformer["n_regimes"])):
        out[f"operating_regime_{r}"] = (regime == r).astype(np.int8)
    group_stats = transformer["group_stats"]
    subset_stats = transformer["subset_stats"]
    global_stats = transformer["global_stats"]
    positions = {}
    for pos, key in enumerate(zip(out["subset"].astype(str), regime)):
        positions.setdefault((key[0], int(key[1])), []).append(pos)
    for source_col in transformer["source_columns"]:
        means = np.empty(len(out), dtype=float)
        stds = np.empty(len(out), dtype=float)
        for key, pos in positions.items():
            stats = group_stats.get(key, subset_stats.get(key[0], global_stats))
            mean, sd = stats.get(source_col, global_stats[source_col])
            means[pos] = mean
            stds[pos] = max(float(sd), 1e-6)
        out[f"{source_col}_regime_z"] = (out[source_col].to_numpy(dtype=float) - means) / stds
    return out

def risk_band(rul, critical, watch):
    if rul <= critical:
        return "Critical"
    if rul <= watch:
        return "Watch"
    return "Normal"


st.title("🛠️ Engine RUL Intelligence")
st.caption("Pooled XGBoost RUL regression · NASA C-MAPSS FD001–FD004")
st.info("Educational predictive-maintenance demo using simulated turbofan trajectories. RUL is measured in operating cycles, not hours. This is not a certified aviation-maintenance tool.")

try:
    artifact = load_artifact()
    sensor_cols = artifact["sensor_cols"]
    setting_cols = artifact["setting_cols"]
    feature_columns = artifact["feature_columns"]
    window = int(artifact["rolling_window"])
    model = artifact["model"]
    train_rows = int(artifact.get("n_training_rows", 0))
    train_engines = int(artifact.get("n_training_engines", 0))
    frames, rul_frames = [], []
    for subset in SUBSETS:
        test = load_table(f"test_{subset}.txt")
        test["subset"] = subset
        test["engine_key"] = subset + "_" + test["unit_nr"].astype(int).astype(str).str.zfill(3)
        frames.append(test)
        rul_values = pd.read_csv(locate_file(f"RUL_{subset}.txt"), header=None).iloc[:, 0].to_numpy(dtype=float)
        ids = np.sort(test["unit_nr"].unique().astype(int))
        if len(ids) != len(rul_values):
            raise ValueError(f"{subset}: {len(ids)} test engines but {len(rul_values)} RUL labels.")
        rul_frames.append(pd.DataFrame({"subset":subset, "unit_nr":ids, "engine_key":[f"{subset}_{u:03d}" for u in ids], "benchmark_true_rul_cycles":rul_values}))
    test_raw = pd.concat(frames, ignore_index=True)
    truth = pd.concat(rul_frames, ignore_index=True)
except Exception as exc:
    st.error(str(exc))
    st.markdown("Run notebooks in order: `01_data_understanding.ipynb`, `02_xgboost_rul_training.ipynb`, `04_robustness_experiments.ipynb`, then `03_test_evaluation_and_explainability.ipynb`.")
    st.stop()

with st.sidebar:
    st.header("Maintenance policy")
    critical_threshold = st.slider("Critical if predicted RUL ≤", min_value=5, max_value=40, value=15, step=1)
    watch_threshold = st.slider("Watch if predicted RUL ≤", min_value=critical_threshold + 1, max_value=80, value=max(30, critical_threshold + 1), step=1)
    st.caption("Configurable demonstration thresholds—not learned probabilities or an approved maintenance policy.")
    subset_filter = st.multiselect("Subsets to display", SUBSETS, default=SUBSETS)
    if not subset_filter:
        subset_filter = SUBSETS

all_features = engineer_features(test_raw, sensor_cols, setting_cols, window, SUBSETS)
if artifact.get("feature_mode", "baseline") == "regime_aware":
    all_features = add_regime_features_for_artifact(all_features, artifact["regime_transformer"])
missing_features = [column for column in feature_columns if column not in all_features.columns]
if missing_features:
    st.error(f"Feature pipeline does not match model artifact. Missing columns: {missing_features[:10]}")
    st.stop()
all_features["predicted_rul_cycles"] = np.maximum(0.0, model.predict(all_features[feature_columns]))
last_rows = (
    all_features.sort_values(["subset", "unit_nr", "time_cycles"])
    .groupby(["subset", "unit_nr"], as_index=False, sort=False).tail(1)
    .sort_values(["subset", "unit_nr"]).reset_index(drop=True)
    .merge(truth, on=["subset", "unit_nr", "engine_key"], how="left", validate="one_to_one")
)
if last_rows["benchmark_true_rul_cycles"].isna().any():
    st.error("Some test-engine RUL labels did not align. Rerun Notebook 01 and check the dataset files.")
    st.stop()
last_rows["absolute_error_uncapped_cycles"] = np.abs(last_rows["predicted_rul_cycles"] - last_rows["benchmark_true_rul_cycles"])
last_rows["risk_band"] = last_rows["predicted_rul_cycles"].map(lambda x: risk_band(float(x), critical_threshold, watch_threshold))
last_cycle_map = test_raw.groupby(["subset", "unit_nr"])["time_cycles"].max().to_dict()
truth_map = {(r.subset, int(r.unit_nr)): float(r.benchmark_true_rul_cycles) for r in truth.itertuples()}
all_features["last_observed_cycle"] = [last_cycle_map[(s, int(u))] for s, u in zip(all_features["subset"], all_features["unit_nr"])]
all_features["benchmark_true_rul_cycles"] = [truth_map[(s, int(u))] + last_cycle_map[(s, int(u))] - float(c) for s, u, c in zip(all_features["subset"], all_features["unit_nr"], all_features["time_cycles"])]
all_features["risk_band"] = all_features["predicted_rul_cycles"].map(lambda x: risk_band(float(x), critical_threshold, watch_threshold))

view_rows = last_rows.loc[last_rows["subset"].isin(subset_filter)].copy()
critical_count = int((view_rows["risk_band"] == "Critical").sum())
col1, col2, col3, col4 = st.columns(4)
col1.metric("Test engines in view", f"{len(view_rows):,}")
col2.metric("Median estimated RUL", f"{view_rows['predicted_rul_cycles'].median():.1f} cycles")
col3.metric("Critical by selected policy", str(critical_count))
col4.metric("MAE vs uncapped benchmark", f"{view_rows['absolute_error_uncapped_cycles'].mean():.1f} cycles")
st.caption(f"Loaded deployment artifact: {train_engines:,} training engines / {train_rows:,} rows across all four training subsets. The benchmark truth and MAE shown here use offline C-MAPSS test labels; they are for retrospective demonstration only, not live predictions or further model selection.")

tab_fleet, tab_engine, tab_model = st.tabs(["Fleet overview", "Engine explorer", "Model & methodology"])
with tab_fleet:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Engine maintenance queue")
        display_cols = view_rows[["subset", "unit_nr", "time_cycles", "predicted_rul_cycles", "risk_band", "benchmark_true_rul_cycles", "absolute_error_uncapped_cycles"]].rename(columns={
            "subset":"Subset", "unit_nr":"Engine", "time_cycles":"Last cycle", "predicted_rul_cycles":"Predicted RUL (cycles)",
            "risk_band":"Policy band", "benchmark_true_rul_cycles":"Benchmark truth (cycles)", "absolute_error_uncapped_cycles":"Absolute error (cycles)"
        }).sort_values("Predicted RUL (cycles)")
        st.dataframe(display_cols.round(2), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Predicted RUL distribution")
        fig = px.histogram(view_rows, x="predicted_rul_cycles", color="subset", nbins=24,
                           labels={"predicted_rul_cycles":"Predicted RUL (cycles)","subset":"Dataset subset"}, title="Test-engine RUL estimates")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Predicted vs benchmark RUL")
        fig2 = px.scatter(view_rows, x="benchmark_true_rul_cycles", y="predicted_rul_cycles", color="subset",
                          hover_data=["unit_nr", "time_cycles", "absolute_error_uncapped_cycles"],
                          labels={"benchmark_true_rul_cycles":"Benchmark RUL (cycles)","predicted_rul_cycles":"Predicted RUL (cycles)","subset":"Subset"})
        upper = float(max(view_rows["benchmark_true_rul_cycles"].max(), view_rows["predicted_rul_cycles"].max(), 1))
        fig2.add_shape(type="line", x0=0, y0=0, x1=upper, y1=upper, line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig2, use_container_width=True)

with tab_engine:
    st.subheader("Engine-level RUL history")
    options = view_rows.sort_values(["subset", "unit_nr"])
    option_labels = {f"{r.subset} · Engine {int(r.unit_nr):03d}": (r.subset, int(r.unit_nr)) for r in options.itertuples()}
    chosen_label = st.selectbox("Choose a test engine", list(option_labels.keys()), index=0)
    chosen_subset, chosen_unit = option_labels[chosen_label]
    unit_history = all_features.loc[(all_features["subset"] == chosen_subset) & (all_features["unit_nr"] == chosen_unit)].sort_values("time_cycles")
    final_row = last_rows.loc[(last_rows["subset"] == chosen_subset) & (last_rows["unit_nr"] == chosen_unit)].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated RUL at last cycle", f"{final_row['predicted_rul_cycles']:.1f} cycles")
    m2.metric("Benchmark RUL (offline truth)", f"{final_row['benchmark_true_rul_cycles']:.1f} cycles")
    m3.metric("Policy band", str(final_row["risk_band"]))
    st.caption("The benchmark line at earlier observed cycles is reconstructed from the official last-cycle label. It is available only for offline evaluation.")
    history_long = unit_history[["time_cycles", "predicted_rul_cycles", "benchmark_true_rul_cycles"]].melt(id_vars="time_cycles", var_name="series", value_name="RUL_cycles")
    history_long["series"] = history_long["series"].map({"predicted_rul_cycles":"XGBoost prediction", "benchmark_true_rul_cycles":"Benchmark reference (offline)"})
    fig = px.line(history_long, x="time_cycles", y="RUL_cycles", color="series", labels={"time_cycles":"Observed operating cycle","RUL_cycles":"RUL (cycles)","series":"Series"}, title=f"{chosen_subset} engine {chosen_unit}: predicted RUL over observed history")
    st.plotly_chart(fig, use_container_width=True)
    available_sensors = [c for c in sensor_cols if c in test_raw.columns]
    selected_sensors = st.multiselect("Sensor channels to inspect", available_sensors, default=available_sensors[:4], key="selected_sensors_all_subset")
    if selected_sensors:
        sensor_history = test_raw.loc[(test_raw["subset"] == chosen_subset) & (test_raw["unit_nr"] == chosen_unit), ["time_cycles"] + selected_sensors].melt(id_vars="time_cycles", var_name="Sensor", value_name="Measurement")
        fig = px.line(sensor_history, x="time_cycles", y="Measurement", color="Sensor", labels={"time_cycles":"Operating cycle","Measurement":"Sensor reading"}, title=f"{chosen_subset} engine {chosen_unit}: selected sensor readings")
        st.plotly_chart(fig, use_container_width=True)

with tab_model:
    st.subheader("Model facts")
    facts = {
        "Algorithm":"XGBoost Regressor (only predictive model family)",
        "Training dataset":artifact.get("dataset", "NASA C-MAPSS FD001-FD004 pooled model"),
        "Target":"Remaining useful life (operating cycles)",
        "Training engines":f"{train_engines:,}",
        "Training rows":f"{train_rows:,}",
        "Training RUL target cap":("Uncapped" if artifact.get("rul_cap") is None else f"{artifact.get('rul_cap')} cycles"),
        "Feature strategy":artifact.get("feature_mode", "baseline"),
        "Training objective":artifact.get("selected_objective", artifact.get("best_params", {}).get("objective", "reg:squarederror")),
        "Sample weighting":artifact.get("sample_weight_policy", "row_uniform"),
        "Trailing window":f"{window} cycles",
        "Sensor channels retained":len(sensor_cols),
        "Operating settings retained":len(setting_cols),
        "Model feature count":len(feature_columns),
        "Subset indicators":", ".join(SUBSETS),
        "Random state":artifact.get("random_state",42),
    }
    st.table(pd.DataFrame(list(facts.items()), columns=["Item","Value"]))
    st.subheader("Top XGBoost feature importances")
    importance = pd.DataFrame({"feature":feature_columns,"importance":model.feature_importances_}).sort_values("importance",ascending=False).head(18).sort_values("importance")
    fig = px.bar(importance, x="importance", y="feature", orientation="h", labels={"importance":"XGBoost importance","feature":"Feature"})
    st.plotly_chart(fig, use_container_width=True)
    metrics_path = REPORTS_DIR / "all_subsets_test_metrics.json"
    if metrics_path.exists():
        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        saved_cap = saved_metrics.get("training_target_cap_cycles", "unknown")
        saved_mode = saved_metrics.get("training_feature_mode", "unknown")
        current_cap = artifact.get("rul_cap", None)
        current_cap_label = "uncapped" if current_cap is None else str(current_cap)
        saved_cap_label = "uncapped" if saved_cap is None else str(saved_cap)
        cap_matches = str(saved_cap_label) == str(current_cap_label)
        mode_matches = str(saved_mode) == str(artifact.get("feature_mode", "baseline"))

        st.subheader("Saved Notebook 03 test-evaluation report")
        if cap_matches and mode_matches:
            st.info(
                f"Stored report configuration matches the loaded artifact (target cap: {current_cap_label}; "
                f"feature mode: {saved_mode}). These test-set labels have already been inspected during project development, "
                "so use validation—not this report—for any further model selection."
            )
        else:
            st.warning(
                "HISTORICAL REPORT — this JSON was generated for a different model configuration and must not be presented "
                "as the evaluation of the currently loaded artifact. "
                f"Stored report: target cap={saved_cap_label}, feature mode={saved_mode}. "
                f"Loaded artifact: target cap={current_cap_label}, feature mode={artifact.get('feature_mode', 'baseline')}. "
                "Do not use these historical test metrics to select or tune the current model."
            )
        with st.expander("View stored Notebook 03 report"):
            st.json(saved_metrics)
    else:
        st.info("No saved Notebook 03 metrics report was found. The dashboard still computes the displayed offline benchmark comparison from the currently loaded artifact and the bundled C-MAPSS test labels.")
    st.markdown("**Methodology notes:** complete run-to-failure training trajectories are pooled across FD001–FD004. Rolling features are engine-grouped, and validation splits are made by engine. The C-MAPSS test labels are exposed here only for retrospective offline demonstration and have already been inspected during development; avoid using them for further model selection. The data is simulated; these results do not establish readiness for real aircraft maintenance.")

st.divider()
st.caption("Portfolio project · Pooled XGBoost-only RUL regression · NASA PCoE C-MAPSS FD001–FD004")
