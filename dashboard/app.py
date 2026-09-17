"""Streamlit entry point for AdoptAI V1 live monitoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.collector_control import CollectorControlError, CollectorController
from dashboard.config import (
    DATABASE_PATH, FINAL_TEST_METRICS_PATH, HISTORY_MINUTES, MODEL_METADATA_PATH,
    MODEL_PATH, PREPROCESSOR_PATH, REFRESH_SECONDS,
)
from dashboard.history import (
    initialize_dashboard_tables, list_runs, load_alerts, load_metrics_history,
    load_prediction_history, log_prediction, session_sample_count,
)
from dashboard.inference import InferenceError, load_inference_engine
from dashboard.live_features import LiveFeatureError, load_recent_run_history, prepare_current_feature_row
from dashboard.ui_components import (
    inject_styles,
    limitation_notice,
    metric_card,
    product_header,
    recording_indicator,
    risk_card,
    short_id,
    status_strip,
)


st.set_page_config(page_title="AdoptAI · Live Monitor", page_icon="A", layout="wide", initial_sidebar_state="collapsed")


def initialize_theme() -> str:
    requested = str(st.query_params.get("theme", "dark")).lower()
    requested = requested if requested in {"dark", "light"} else "dark"
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = requested
    if "light_theme" not in st.session_state:
        st.session_state.light_theme = st.session_state.ui_theme == "light"
    return str(st.session_state.ui_theme)


def sync_theme_preference() -> None:
    selected = "light" if st.session_state.light_theme else "dark"
    st.session_state.ui_theme = selected
    st.query_params["theme"] = selected


theme = initialize_theme()
inject_styles(st, PROJECT_ROOT / "dashboard/styles.css", theme)
header_main, header_theme = st.columns([5.5, 1.15], vertical_alignment="center")
with header_main:
    product_header(st)
with header_theme:
    st.markdown('<div class="theme-control-label">Appearance</div>', unsafe_allow_html=True)
    st.toggle("Light mode", key="light_theme", on_change=sync_theme_preference)


@st.cache_resource(show_spinner=False)
def inference_engine():
    return load_inference_engine()


@st.cache_resource(show_spinner=False)
def collector_controller():
    return CollectorController(DATABASE_PATH)


def local_time(timestamp: str | None) -> str:
    if not timestamp:
        return "—"
    value = pd.to_datetime(timestamp, errors="coerce", utc=True)
    return "—" if pd.isna(value) else value.tz_convert("Africa/Casablanca").strftime("%H:%M:%S")


CHART_COLORS = {
    "CPU": "#22B8C3",
    "RAM": "#7C70E8",
    "Swap": "#D99827",
    "Disk": "#DF7B27",
    "Risk Score": "#D95361",
}


def themed_line_chart(
    frame: pd.DataFrame,
    timestamp_column: str,
    series: dict[str, str],
    active_theme: str,
    y_title: str,
    fixed_domain: list[float] | None = None,
) -> None:
    values = frame[[timestamp_column, *series]].rename(columns=series).melt(
        id_vars=timestamp_column, var_name="Series", value_name="Value"
    ).dropna(subset=[timestamp_column, "Value"])
    if values.empty:
        st.caption("No values are available in this time range.")
        return
    light = active_theme == "light"
    text_color = "#526176" if light else "#9EB0C5"
    grid_color = "rgba(35,55,80,.10)" if light else "rgba(158,176,197,.13)"
    domain = list(series.values())
    chart = (
        alt.Chart(values)
        .mark_line(strokeWidth=2, interpolate="monotone")
        .encode(
            x=alt.X(f"{timestamp_column}:T", title=None, axis=alt.Axis(format="%H:%M")),
            y=alt.Y("Value:Q", title=y_title, scale=alt.Scale(domain=fixed_domain, zero=False)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=domain, range=[CHART_COLORS[name] for name in domain]),
                legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip(f"{timestamp_column}:T", title="Time"),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Value:Q", format=".2f"),
            ],
        )
        .properties(height=230)
        .configure(background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor=text_color, titleColor=text_color, gridColor=grid_color, domainColor=grid_color)
        .configure_legend(labelColor=text_color)
    )
    st.altair_chart(chart, width="stretch", theme=None)


def live_charts(history: pd.DataFrame, predictions: pd.DataFrame, active_theme: str) -> None:
    if history.empty:
        st.info("Live charts will appear after measurements arrive.")
        return
    cutoff = history.timestamp.max() - pd.Timedelta(minutes=HISTORY_MINUTES)
    recent = history.loc[history.timestamp.ge(cutoff)]
    left, middle, right = st.columns(3)
    with left:
        st.markdown("#### CPU & RAM")
        themed_line_chart(recent, "timestamp", {"cpu_pct": "CPU", "ram_pct": "RAM"}, active_theme, "Usage (%)", [0, 100])
    with middle:
        st.markdown("#### Swap & disk usage")
        themed_line_chart(recent, "timestamp", {"swap_pct": "Swap", "disk_usage_pct": "Disk"}, active_theme, "Usage (%)", [0, 100])
    with right:
        st.markdown("#### Risk Score")
        if predictions.empty:
            st.caption("Available after model warm-up.")
        else:
            themed_line_chart(predictions, "timestamp_utc", {"risk_score": "Risk Score"}, active_theme, "Score", [0, 100])


def render_live_monitor(active_theme: str) -> None:
    controller = collector_controller()
    status = controller.status()
    status_strip(st, status.state, status.machine_id, status.run_id, local_time(status.last_sample_at))
    _, sample_count = session_sample_count(DATABASE_PATH, status.run_id)
    recording_indicator(st, status.state, sample_count, DATABASE_PATH.name)
    start_col, stop_col, detail_col = st.columns([1, 1, 5])
    with start_col:
        if st.button(
            "Start Collection", key="start_collection", type="primary" if not status.running else "secondary",
            width="stretch", disabled=status.running,
        ):
            try:
                controller.start(); st.rerun()
            except Exception as exc:
                st.error(f"Collection could not start: {exc}")
    with stop_col:
        can_stop = status.running and status.owned
        if st.button(
            "Stop Collection", key="stop_collection", type="primary" if can_stop else "secondary",
            width="stretch", disabled=not can_stop,
        ):
            try:
                controller.stop(); st.rerun()
            except CollectorControlError as exc:
                st.error(str(exc))
    with detail_col:
        st.caption(status.message)

    if status.state == "Error":
        st.error(status.message)
    if status.running and not status.owned:
        st.warning("An exact external collector is connected read-only. Stop it from the process that started it.")
    if not status.run_id:
        st.markdown("<div class='surface-card'><strong>Ready to monitor</strong><div class='fine-print'>Start a collection session to create a new run and begin live measurements.</div></div>", unsafe_allow_html=True)
        risk_card(st, None)
        return

    try:
        engine = inference_engine()
        raw_history = load_recent_run_history(DATABASE_PATH, status.run_id)
        if raw_history.empty:
            st.info("Collector started. Waiting for the first stored measurement…")
            risk_card(st, None)
            return
        readiness = prepare_current_feature_row(raw_history, engine.input_feature_names)
        latest = raw_history.iloc[-1]
        result = engine.predict(readiness.feature_row) if readiness.ready and readiness.feature_row is not None else None
        if result is not None:
            log_prediction(
                DATABASE_PATH, str(latest.timestamp), str(latest.machine_id), str(latest.run_id), result,
                float(latest.cpu_pct) if pd.notna(latest.cpu_pct) else None,
                float(latest.ram_pct) if pd.notna(latest.ram_pct) else None,
            )
        risk_col, metrics_col = st.columns([1.12, 1.88], gap="large")
        with risk_col:
            risk_card(st, result)
            if not readiness.ready:
                st.markdown("<div class='warm-card'><strong>Model warming up</strong><br><span class='fine-print'>Collecting enough recent system history for prediction…</span></div>", unsafe_allow_html=True)
                st.progress(readiness.progress, text=f"{readiness.history_seconds:.0f} of 120 continuous seconds")
            elif result and result.predicted_class == 1:
                st.markdown(f"<div class='alert-card'><strong>Slowdown risk detected for the next 5 minutes.</strong><br>Risk Score {result.risk_score:.0f}/100 · CPU {latest.cpu_pct:.1f}% · RAM {latest.ram_pct:.1f}%</div>", unsafe_allow_html=True)
        with metrics_col:
            st.markdown("### Current computer metrics")
            cards = st.columns(4)
            values = [
                ("CPU usage", latest.cpu_pct, "%"), ("RAM usage", latest.ram_pct, "%"),
                ("Swap usage", latest.swap_pct, "%"), ("Disk usage", latest.disk_usage_pct, "%"),
                ("Disk latency", latest.disk_latency_ms, "ms"), ("Context switches", latest.context_switches_per_s, "/s"),
                ("Processes", int(latest.process_count) if pd.notna(latest.process_count) else None, ""),
                ("Threads", int(latest.thread_count) if pd.notna(latest.thread_count) else None, ""),
            ]
            for index, (label, value, unit) in enumerate(values):
                with cards[index % 4]: metric_card(st, label, float(value) if pd.notna(value) else None, unit)
        metrics = load_metrics_history(DATABASE_PATH, status.run_id, limit=1_000)
        prediction_history = load_prediction_history(DATABASE_PATH, status.run_id, limit=1_000)
        st.markdown("### Recent telemetry")
        live_charts(metrics, prediction_history, active_theme)
        alerts = load_alerts(DATABASE_PATH, status.run_id, limit=8)
        st.markdown("### Recent alerts")
        if alerts.empty:
            st.caption("No slowdown-risk alerts have been logged for this session.")
        else:
            shown = alerts[["timestamp_utc", "risk_score", "status", "cpu_pct", "ram_pct"]].rename(columns={"timestamp_utc":"Time","risk_score":"Risk Score","status":"Status","cpu_pct":"CPU %","ram_pct":"RAM %"})
            st.dataframe(shown, width="stretch", hide_index=True)
    except (LiveFeatureError, InferenceError, OSError, ValueError) as exc:
        st.error(f"Live inference is unavailable: {exc}")


def render_history(active_theme: str) -> None:
    st.markdown("## Session history")
    runs = list_runs(DATABASE_PATH)
    if runs.empty:
        st.info("No collection sessions are available yet.")
        return
    labels = {
        row.run_id: (
            f"{pd.to_datetime(row.started_at_utc, utc=True).tz_convert('Africa/Casablanca').strftime('%d %b %Y · %H:%M')}"
            f" · Session {short_id(row.run_id, 8)} · {str(row.status).capitalize()}"
        )
        for row in runs.itertuples()
    }
    run_id = st.selectbox("Monitoring session", runs.run_id.tolist(), format_func=lambda value: labels[value])
    st.caption(f"Full run ID: `{run_id}`")
    metrics = load_metrics_history(DATABASE_PATH, run_id)
    predictions = load_prediction_history(DATABASE_PATH, run_id)
    alerts = load_alerts(DATABASE_PATH, run_id, limit=100)
    if metrics.empty:
        st.info("This session has no metric rows.")
        return
    minimum, maximum = metrics.timestamp.min().to_pydatetime(), metrics.timestamp.max().to_pydatetime()
    selected = st.slider("Timestamp range", min_value=minimum, max_value=maximum, value=(minimum, maximum), format="DD MMM HH:mm:ss") if maximum > minimum else (minimum, maximum)
    filtered = metrics[metrics.timestamp.between(pd.Timestamp(selected[0]), pd.Timestamp(selected[1]))]
    filtered_predictions = predictions[predictions.timestamp_utc.between(pd.Timestamp(selected[0]), pd.Timestamp(selected[1]))] if not predictions.empty else predictions
    a,b,c,d = st.columns(4)
    a.metric("Measurements", f"{len(filtered):,}"); b.metric("Predictions", f"{len(filtered_predictions):,}"); c.metric("Alerts", f"{len(alerts):,}"); d.metric("Session", short_id(run_id))
    live_charts(filtered, filtered_predictions, active_theme)
    st.markdown("### Prediction history")
    if filtered_predictions.empty: st.caption("No model scores were recorded for this session.")
    else: st.dataframe(filtered_predictions[["timestamp_utc","risk_score","predicted_class","status","cpu_pct","ram_pct"]].sort_values("timestamp_utc",ascending=False),width="stretch",hide_index=True)
    st.markdown("### Alert history")
    if alerts.empty: st.caption("No alerts were recorded for this session.")
    else: st.dataframe(alerts[["timestamp_utc","risk_score","status","cpu_pct","ram_pct","alert_reason"]],width="stretch",hide_index=True)


def render_model_info() -> None:
    st.markdown("## Model information")
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    metrics = pd.read_csv(FINAL_TEST_METRICS_PATH).iloc[0]
    left,right = st.columns([1.25,1],gap="large")
    with left:
        st.markdown("<div class='surface-card'><div class='eyebrow'>Experimental prototype</div><div class='metric-value' style='font-size:2.2rem'>LightGBM V1</div><p class='fine-print'>Predicts slowdown risk during the next 5 minutes.</p></div>",unsafe_allow_html=True)
        limitation_notice(st)
        st.markdown("### Risk Score")
        st.write("Risk Score represents the model's relative slowdown-risk signal for the next 5 minutes. It is not a calibrated probability.")
        st.markdown("**Risk Score = raw LightGBM score × 100**")
        st.caption("Prototype reference boundary: **50 / 100**")
    with right:
        st.markdown("### Final independent test")
        m1,m2=st.columns(2); m1.metric("PR-AUC",f"{metrics.pr_auc:.3f}"); m2.metric("ROC-AUC",f"{metrics.roc_auc:.3f}")
        m3,m4,m5=st.columns(3); m3.metric("Precision",f"{metrics.precision:.3f}"); m4.metric("Recall",f"{metrics.recall:.3f}"); m5.metric("F1",f"{metrics.f1:.3f}")
    with st.expander("Technical details", expanded=False):
        st.code(
            f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)}\n"
            f"Preprocessor: {PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)}\n"
            f"Inputs: 228 → transformed: 360\n"
            f"Internal model status: {metadata['model_role']}\n"
            "Frozen binary threshold: 0.50",
            language="text",
        )
    st.markdown("### Appropriate use")
    st.write("Use this dashboard for monitoring, demonstration, and collecting additional representative operating-regime data. Do not treat its alerts as guaranteed diagnoses or production safety controls.")


initialize_dashboard_tables(DATABASE_PATH)
live_tab, history_tab, model_tab = st.tabs(["Live Monitor", "History", "Model Info"])
with live_tab:
    if hasattr(st, "fragment"):
        st.fragment(run_every=REFRESH_SECONDS)(render_live_monitor)(theme)
    else:
        render_live_monitor(theme)
with history_tab: render_history(theme)
with model_tab: render_model_info()
