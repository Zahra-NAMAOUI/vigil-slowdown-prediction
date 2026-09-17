"""Reusable polished Streamlit presentation components."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


THEME_TOKENS = {
    "dark": {
        "background": "#07111F",
        "surface": "#0D1A2B",
        "surface_secondary": "#112238",
        "text_primary": "#F4F8FC",
        "text_secondary": "#9EB0C5",
        "text_muted": "#70839A",
        "border": "rgba(158, 176, 197, .16)",
        "header": "rgba(7, 17, 31, .88)",
        "shadow": "0 12px 38px rgba(0, 0, 0, .16)",
        "chart_grid": "rgba(158, 176, 197, .13)",
        "accent_soft": "rgba(53, 198, 208, .10)",
        "warning_soft": "rgba(231, 184, 75, .08)",
        "danger_soft": "rgba(239, 102, 115, .09)",
        "warning_text": "#F4DFAD",
        "danger_text": "#FFD8DC",
    },
    "light": {
        "background": "#F7F9FC",
        "surface": "#FFFFFF",
        "surface_secondary": "#F1F5F9",
        "text_primary": "#142033",
        "text_secondary": "#526176",
        "text_muted": "#718096",
        "border": "rgba(35, 55, 80, .13)",
        "header": "rgba(247, 249, 252, .92)",
        "shadow": "0 10px 30px rgba(20, 32, 51, .07)",
        "chart_grid": "rgba(35, 55, 80, .10)",
        "accent_soft": "rgba(18, 166, 177, .10)",
        "warning_soft": "rgba(180, 120, 15, .09)",
        "danger_soft": "rgba(202, 60, 76, .08)",
        "warning_text": "#76530C",
        "danger_text": "#9E2F3B",
    },
}


def theme_css(theme: str) -> str:
    """Return shared CSS variables for one validated presentation theme."""
    selected = theme if theme in THEME_TOKENS else "dark"
    tokens = THEME_TOKENS[selected]
    variables = "\n".join(
        f"--{name.replace('_', '-')}: {value};" for name, value in tokens.items()
    )
    return f"""
    :root {{
      color-scheme: {selected};
      {variables}
      --accent: #35C6D0;
      --accent-strong: #0D96A1;
      --success: #2FA875;
      --warning: #D99827;
      --danger: #D95361;
      --violet: #7C70E8;
    }}
    """


def inject_styles(st: Any, css_path: Path, theme: str = "dark") -> None:
    shared = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{shared}\n{theme_css(theme)}</style>", unsafe_allow_html=True)


def short_id(value: str | None, length: int = 10) -> str:
    if not value:
        return "—"
    text = str(value)
    return text if len(text) <= length else f"{text[:length]}…"


def product_header(st: Any) -> None:
    st.markdown(
        """
        <div class="brand-shell">
          <div class="brand-row">
            <div class="brand-mark">A</div>
            <div class="brand-name">AdoptAI</div>
          </div>
          <p class="subtitle">Intelligent Computer Performance Monitor</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recording_indicator(
    st: Any,
    state: str,
    sample_count: int,
    database_name: str,
) -> None:
    running = state == "Running"
    starting = state == "Starting"
    if running:
        label = "Recording telemetry"
        detail = f"Samples this session: {sample_count:,}"
        css_class = "recording-live"
    elif starting:
        label = "Preparing telemetry recording"
        detail = "Waiting for the first stored sample"
        css_class = "recording-starting"
    else:
        label = "Recording stopped"
        detail = f"Last session samples: {sample_count:,}" if sample_count else "No session samples yet"
        css_class = ""
    st.markdown(
        f"""<div class="recording-strip {css_class}">
        <div><span class="recording-dot"></span><strong>{html.escape(label)}</strong></div>
        <div class="recording-meta">Storage: {html.escape(database_name)}<span></span>{html.escape(detail)}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def status_strip(st: Any, state: str, machine_id: str | None, run_id: str | None, last_update: str | None) -> None:
    state_class = {
        "Running": "state-running", "Starting": "state-starting", "Error": "state-error"
    }.get(state, "")
    items = [
        ("Collector", f'<span class="state-dot {state_class}"></span>{html.escape(state)}'),
        ("Machine", html.escape(short_id(machine_id, 14))),
        ("Session", html.escape(short_id(run_id, 14))),
        ("Last update", html.escape(last_update or "—")),
    ]
    body = "".join(
        f'<div class="status-item"><div class="eyebrow">{label}</div><div class="status-value" title="{html.escape(str(value))}">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="status-strip">{body}</div>', unsafe_allow_html=True)


def metric_card(st: Any, label: str, value: float | int | None, unit: str, note: str = "Live measurement") -> None:
    shown = "—" if value is None else f"{value:,.1f}" if isinstance(value, float) else f"{value:,}"
    st.markdown(
        f"""<div class="metric-card"><div class="eyebrow">{html.escape(label)}</div>
        <div class="metric-value">{shown} <span style="font-size:.85rem;color:var(--muted)">{html.escape(unit)}</span></div>
        <div class="metric-note">{html.escape(note)}</div></div>""",
        unsafe_allow_html=True,
    )


def risk_card(st: Any, result: Any | None) -> None:
    def scale(marker: float | None) -> str:
        marker_html = "" if marker is None else (
            f'<span class="risk-marker" style="left:{max(0.0, min(100.0, marker)):.2f}%" '
            f'aria-label="Current Risk Score {marker:.1f}"></span>'
        )
        return f"""<div class="risk-scale" aria-label="Risk presentation scale">
          <div class="risk-track">{marker_html}</div>
          <div class="risk-ticks"><span>0</span><span>30</span><span>50</span><span>70</span><span>100</span></div>
          <div class="risk-bands"><span>Low</span><span>Moderate</span><span>Warning</span><span>High</span></div>
        </div>"""

    if result is None:
        st.markdown(
            """<div class="risk-card"><div class="eyebrow">Model Risk Score</div>
            <div class="risk-score">— <span class="risk-denominator">/ 100</span></div>
            <span class="risk-level">Unavailable</span>
            """ + scale(None) + """
            <div class="risk-caption">Prediction horizon: next 5 minutes. A score appears only after the required continuous history is available.</div></div>""",
            unsafe_allow_html=True,
        )
        return
    css_status = result.status.lower().replace(" ", "-")
    st.markdown(
        f"""<div class="risk-card"><div class="eyebrow">Model Risk Score</div>
        <div class="risk-score">{result.risk_score:.0f} <span class="risk-denominator">/ 100</span></div>
        <span class="risk-level risk-{css_status}">{html.escape(result.status)}</span>
        {scale(result.risk_score)}
        <div class="risk-caption">Prediction horizon: next 5 minutes · Raw model score {result.model_score:.3f}. This is a relative model score, not a calibrated probability.</div></div>""",
        unsafe_allow_html=True,
    )


def limitation_notice(st: Any) -> None:
    st.markdown(
        """<div class="limitation"><strong>Experimental prototype</strong><br>
        Performance may vary across machines and operating regimes because distribution shift was observed during evaluation.</div>""",
        unsafe_allow_html=True,
    )
