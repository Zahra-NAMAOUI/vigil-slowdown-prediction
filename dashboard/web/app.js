/* Vigil — front end. Vanilla JS, no build step.
   Talks only to the FastAPI layer in dashboard/api.py. Nothing here computes a
   score. Endpoints, fields and displayed values are unchanged from the previous
   version: this file differs only in how it lays the same data out. */

'use strict';

const LIVE_INTERVAL_MS = 2000;
const HISTORY_INTERVAL_MS = 10000;
const SCORE_ANIMATION_MS = 200;
const GAUGE_CIRCUMFERENCE = 565.487;   // 2 * pi * 90
const GAUGE_ARC = 424.115;             // 270 degrees of it
const DRIVER_SIGMA_CAP = 3;            // bars saturate at 3 sigma

/* Band colours follow the status string returned by the backend
   (dashboard/inference.py risk_status). The front end never recomputes a band. */
const BAND_COLORS = {
  'Low': '#35C6D0',        // below the alert boundary → accent teal
  'Moderate': '#35C6D0',   // still below the alert boundary → accent teal
  'Warning': '#D99827',    // warning
  'High Risk': '#D95361',  // danger
};

const COLLECTOR_STATE_TEXT = {
  'Running': 'Monitoring live',
  'Starting': 'Starting collector',
  'Stopped': 'Monitoring stopped',
  'Error': 'Collector error',
};

/* `max` gives the tile its progress bar. Metrics with no bounded scale get no
   bar rather than an invented one. */
const TELEMETRY_TILES = [
  { key: 'cpu_pct', label: 'CPU usage', unit: '%', digits: 1, max: 100 },
  { key: 'ram_pct', label: 'RAM usage', unit: '%', digits: 1, max: 100 },
  { key: 'swap_pct', label: 'Swap usage', unit: '%', digits: 1, max: 100 },
  { key: 'disk_usage_pct', label: 'Disk usage', unit: '%', digits: 1, max: 100 },
  { key: 'disk_latency_ms', label: 'Disk latency', unit: 'ms', digits: 2, max: null },
  { key: 'context_switches_per_s', label: 'Context switches', unit: '/s', digits: 0, max: null },
  { key: 'process_count', label: 'Processes', unit: '', digits: 0, max: null },
  { key: 'thread_count', label: 'Threads', unit: '', digits: 0, max: null },
];

const OVERVIEW_TILE_KEYS = [
  'cpu_pct', 'ram_pct', 'swap_pct', 'disk_usage_pct', 'disk_latency_ms', 'process_count',
];

const state = {
  displayedScore: 0,
  animationFrame: null,
  liveSkipTicks: 0,
  historySkipTicks: 0,
  failureStreak: 0,
  liveTimer: null,
  historyTimer: null,
  collectorBusy: false,
  lastCollector: null,
  mode: 'live',          // set once from /api/health; never inferred elsewhere
  page: 'overview',
  riskChart: null,
  metricsChart: null,
};

/* ═══════════ helpers ═══════════ */

const $ = (id) => document.getElementById(id);

function round(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/** Every number is rounded before it reaches the DOM. */
function fmt(value, digits = 0, fallback = '—') {
  const rounded = round(value, digits);
  if (rounded === null) return fallback;
  return rounded.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtPercent(fraction, digits = 1) {
  if (fraction === null || fraction === undefined || Number.isNaN(fraction)) return '—';
  return `${fmt(fraction * 100, digits)}%`;
}

/** Truncate UUIDs visually, preserve the full value in the title attribute. */
function shortId(value, length = 12) {
  if (!value) return '—';
  const text = String(value);
  return text.length <= length ? text : `${text.slice(0, length)}…`;
}

function setIdField(element, value) {
  element.textContent = shortId(value);
  element.title = value ? String(value) : '';
}

function localTime(isoString) {
  if (!isoString) return '—';
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function localDate(isoString) {
  if (!isoString) return '—';
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' });
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (character) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]
  ));
}

async function getJSON(path) {
  const response = await fetch(path, { headers: { Accept: 'application/json' } });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = String(body.detail);
    } catch (_) { /* non-JSON error body */ }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function postJSON(path) {
  const response = await fetch(path, { method: 'POST', headers: { Accept: 'application/json' } });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = new Error(body && body.detail ? String(body.detail) : `${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

/* ═══════════ toast / reconnect indicator ═══════════ */

function showNotice(message, danger = false) {
  $('toastText').textContent = message;
  $('toastDot').className = danger ? 'toast-dot is-danger' : 'toast-dot';
  $('toast').hidden = false;
}

function hideNotice() {
  $('toast').hidden = true;
}

function registerFailure(error) {
  state.failureStreak += 1;
  // Exponential backoff expressed in skipped poll ticks, capped so it always recovers.
  const skip = Math.min(2 ** (state.failureStreak - 1), 15);
  state.liveSkipTicks = skip;
  state.historySkipTicks = Math.ceil(skip / 3);
  const isServerError = Boolean(error && error.status);
  showNotice(isServerError ? `Unavailable — ${error.message}` : 'Reconnecting…', isServerError);
}

function registerSuccess() {
  if (state.failureStreak !== 0) {
    state.failureStreak = 0;
    hideNotice();
  }
}

/* ═══════════ navigation ═══════════ */

function showPage(name) {
  state.page = name;
  for (const button of document.querySelectorAll('.nav-item')) {
    button.classList.toggle('is-active', button.dataset.page === name);
  }
  for (const section of document.querySelectorAll('.page')) {
    section.hidden = section.id !== `page-${name}`;
  }
  // ECharts cannot measure a hidden container, so size them when Overview appears.
  if (name === 'overview' && state.riskChart) {
    state.riskChart.resize();
    state.metricsChart.resize();
  }
}

for (const button of document.querySelectorAll('.nav-item')) {
  button.addEventListener('click', () => showPage(button.dataset.page));
}

/* ═══════════ risk gauge ═══════════ */

function paintGauge(score, color) {
  const clamped = Math.max(0, Math.min(100, score));
  $('gaugeValue').style.strokeDasharray = `${(clamped / 100) * GAUGE_ARC} ${GAUGE_CIRCUMFERENCE}`;
  $('scoreValue').textContent = fmt(clamped, 0);
  $('scoreBar').style.width = `${clamped}%`;
  $('boundaryValue').textContent = `${fmt(clamped, 0)} · boundary 50`;
  if (color) document.documentElement.style.setProperty('--risk-color', color);
}

/** No valid prediction exists. Wipe every score-driven element so nothing stale
    — and nothing derived from warm-up progress — can be read as a score. */
function clearScoreDisplay() {
  if (state.animationFrame) {
    cancelAnimationFrame(state.animationFrame);
    state.animationFrame = null;
  }
  state.displayedScore = 0;
  $('gaugeValue').style.strokeDasharray = `0 ${GAUGE_CIRCUMFERENCE}`;
  $('scoreValue').textContent = '—';
  $('scoreBar').style.width = '0%';
  $('boundaryValue').textContent = '— · boundary 50';
  $('statusPill').textContent = 'No prediction';
  document.documentElement.style.removeProperty('--risk-color');
}

/** Smooth 200ms transition between two scores. requestAnimationFrame, no library. */
function animateScore(nextScore, color) {
  if (state.animationFrame) cancelAnimationFrame(state.animationFrame);
  const from = state.displayedScore;
  const delta = nextScore - from;

  if (Math.abs(delta) < 0.5) {
    state.displayedScore = nextScore;
    paintGauge(nextScore, color);
    return;
  }

  const startedAt = performance.now();
  const step = (now) => {
    const progress = Math.min(1, (now - startedAt) / SCORE_ANIMATION_MS);
    const eased = 1 - (1 - progress) ** 3;
    state.displayedScore = from + delta * eased;
    paintGauge(state.displayedScore, color);
    if (progress < 1) {
      state.animationFrame = requestAnimationFrame(step);
    } else {
      state.displayedScore = nextScore;
      state.animationFrame = null;
    }
  };
  state.animationFrame = requestAnimationFrame(step);
}

/* ═══════════ "why this score" ═══════════ */

/* PROVISIONAL EXPLANATION — SHAP REPLACEMENT POINT.
   The backend ranks raw metrics by their deviation from their own 120-second
   rolling baseline (see _score_drivers in dashboard/api.py). It is a descriptive
   signal, not the model's own attribution. When SHAP lands, /api/live should
   return the same {label, value, unit, digits, z} shape carrying per-feature
   attributions; this renderer needs no change. */
function renderDrivers(drivers) {
  const container = $('drivers');
  const empty = $('driversEmpty');

  if (!drivers || drivers.length === 0) {
    container.innerHTML = '';
    empty.hidden = false;
    return;
  }
  empty.hidden = true;

  container.innerHTML = drivers.map((driver) => {
    const magnitude = Math.min(Math.abs(driver.z), DRIVER_SIGMA_CAP) / DRIVER_SIGMA_CAP;
    const halfWidth = magnitude * 50;
    const above = driver.z >= 0;
    const left = above ? 50 : 50 - halfWidth;
    const color = above ? 'var(--warn)' : 'var(--accent)';
    const sign = above ? '+' : '−';
    const unit = driver.unit ? ` ${escapeHtml(driver.unit)}` : '';
    return `
      <div class="driver-row">
        <div class="driver-label">${escapeHtml(driver.label)}</div>
        <div class="driver-track">
          <div class="driver-bar" style="left:${left}%;width:${halfWidth}%;background:${color}"></div>
        </div>
        <div class="driver-value" title="${sign}${fmt(Math.abs(driver.z), 2)} standard deviations from its 120s baseline">
          ${fmt(driver.value, driver.digits ?? 1)}${unit}
        </div>
      </div>`;
  }).join('');
}

/* ═══════════ live rendering ═══════════ */

function renderCollector(collector, alerting) {
  state.lastCollector = collector;
  const dot = $('statusDot');
  const button = $('collectorBtn');

  dot.className = 'status-dot';
  if (alerting) dot.classList.add('is-alert');
  else if (collector.state === 'Running') dot.classList.add('is-running');
  else if (collector.state === 'Starting') dot.classList.add('is-starting');
  else if (collector.state === 'Error') dot.classList.add('is-error');

  $('monitorState').textContent = (state.mode === 'replay' && collector.running)
    ? 'Replay session'
    : (COLLECTOR_STATE_TEXT[collector.state] || collector.state);
  setIdField($('machineId'), collector.machine_id);
  setIdField($('runId'), collector.run_id);

  if (collector.running) {
    button.textContent = 'Stop collection';
    button.className = 'btn btn-secondary';
    // A collector started outside this dashboard is shown, never stopped from here.
    button.disabled = state.collectorBusy || !collector.owned;
    button.title = collector.owned ? '' : collector.message;
  } else {
    button.textContent = 'Start collection';
    button.className = 'btn btn-primary';
    button.disabled = state.collectorBusy;
    button.title = '';
  }
}

function tileMarkup(tile, value) {
  const unit = tile.unit ? `<span class="tile-unit">${escapeHtml(tile.unit)}</span>` : '';
  const bar = tile.max
    ? `<div class="tile-bar"><span style="width:${Math.max(0, Math.min(100, ((value ?? 0) / tile.max) * 100))}%"></span></div>`
    : '<div class="tile-bar is-unscaled"></div>';
  return `
    <div class="tile">
      <div class="eyebrow">${escapeHtml(tile.label)}</div>
      <div class="tile-value">${fmt(value, tile.digits)}${unit}</div>
      ${bar}
    </div>`;
}

function renderSample(sample, lastSampleAt) {
  const overview = TELEMETRY_TILES.filter((tile) => OVERVIEW_TILE_KEYS.includes(tile.key));
  $('overviewTiles').innerHTML = overview
    .map((tile) => tileMarkup(tile, sample ? sample[tile.key] : null)).join('');
  $('telemetryTiles').innerHTML = TELEMETRY_TILES
    .map((tile) => tileMarkup(tile, sample ? sample[tile.key] : null)).join('');
  $('telemetryStamp').textContent = lastSampleAt ? `Updated ${localTime(lastSampleAt)}` : '—';
}

function renderWarmup(live) {
  const progress = Math.max(0, Math.min(1, live.progress || 0));
  $('warmupArc').style.strokeDasharray = `${progress * GAUGE_ARC} ${GAUGE_CIRCUMFERENCE}`;
  $('warmupBar').style.width = `${progress * 100}%`;
  $('warmupValue').textContent = `${fmt(live.elapsed, 0)}s / ${fmt(live.required, 0)}s`;
  $('warmupHeading').textContent =
    `Model warming up — ${fmt(live.elapsed, 0)}s / ${fmt(live.required, 0)}s`;
  $('warmupReason').textContent = live.reason || '';
}

function renderLive(live) {
  const alerting = Boolean(live.score && live.score.predicted_class === 1);
  renderCollector(live.collector, alerting);
  renderSample(live.sample, live.collector ? live.collector.last_sample_at : null);

  if (live.ready && live.score) {
    $('hero').hidden = false;
    $('warmup').hidden = true;
    const color = BAND_COLORS[live.score.status] || BAND_COLORS.Low;
    $('statusPill').textContent = live.score.status;
    animateScore(live.score.risk_score, color);
    renderDrivers(live.drivers);
  } else {
    // Warm-up is a normal state, not an error: the hero is replaced, not blanked.
    // ready:false means there is no prediction, so the score display is cleared
    // outright — no number, no arc fill, no status band.
    $('hero').hidden = true;
    $('warmup').hidden = false;
    clearScoreDisplay();
    renderWarmup(live);
    renderDrivers([]);
  }
}

/* ═══════════ charts ═══════════ */

const AXIS_STYLE = {
  axisLine: { lineStyle: { color: '#222B27' } },
  axisTick: { show: false },
  axisLabel: { color: '#9BA5A1', fontSize: 11 },
  splitLine: { lineStyle: { color: 'rgba(34,43,39,0.7)' } },
};

const TOOLTIP_STYLE = {
  trigger: 'axis',
  backgroundColor: '#121816',
  borderColor: '#222B27',
  borderWidth: 1,
  textStyle: { color: '#F2F5F4', fontSize: 12 },
  axisPointer: { lineStyle: { color: '#222B27' } },
};

function baseLine(name, color, data, yAxisIndex = 0) {
  return {
    name, type: 'line', data, yAxisIndex,
    showSymbol: false, smooth: true, sampling: 'lttb',
    lineStyle: { width: 2, color }, itemStyle: { color }, areaStyle: null,
  };
}

function initCharts() {
  state.riskChart = echarts.init($('riskChart'), null, { renderer: 'canvas' });
  state.metricsChart = echarts.init($('metricsChart'), null, { renderer: 'canvas' });
  // Shared x axis behaviour: one hover moves both crosshairs.
  state.riskChart.group = 'vigil';
  state.metricsChart.group = 'vigil';
  echarts.connect('vigil');
  window.addEventListener('resize', () => {
    state.riskChart.resize();
    state.metricsChart.resize();
  });
}

function renderCharts(history) {
  const metrics = history.metrics || [];
  const predictions = history.predictions || [];
  $('chartsEmpty').hidden = metrics.length > 0 || predictions.length > 0;

  const riskSeries = predictions.map((row) => [row.timestamp_utc, round(row.risk_score, 1)]);
  const pick = (key, digits) => metrics.map((row) => [row.timestamp, round(row[key], digits)]);

  state.riskChart.setOption({
    animation: false,               // no animation on append: it stutters at 10s cadence
    backgroundColor: 'transparent',
    grid: { left: 52, right: 58, top: 26, bottom: 6, containLabel: false },
    tooltip: TOOLTIP_STYLE,
    legend: {
      data: ['Risk score'], top: 0, right: 0,
      textStyle: { color: '#9BA5A1', fontSize: 11 }, icon: 'roundRect', itemWidth: 10, itemHeight: 3,
    },
    xAxis: { type: 'time', ...AXIS_STYLE, axisLabel: { show: false }, splitLine: { show: false } },
    yAxis: {
      type: 'value', min: 0, max: 100, name: 'Score',
      nameTextStyle: { color: '#9BA5A1', fontSize: 10 }, ...AXIS_STYLE,
    },
    series: [{
      ...baseLine('Risk score', '#D95361', riskSeries),
      markLine: {
        silent: true, symbol: 'none', animation: false,
        label: { formatter: 'Alert boundary 50', color: '#9BA5A1', fontSize: 10, position: 'insideEndTop' },
        lineStyle: { color: '#9BA5A1', type: 'dashed', width: 1 },
        data: [{ yAxis: 50 }],
      },
    }],
  }, { notMerge: true, lazyUpdate: false });

  state.metricsChart.setOption({
    animation: false,
    backgroundColor: 'transparent',
    grid: { left: 52, right: 58, top: 26, bottom: 30, containLabel: false },
    tooltip: TOOLTIP_STYLE,
    legend: {
      data: ['CPU', 'RAM', 'Swap', 'Disk latency'], top: 0, right: 0,
      textStyle: { color: '#9BA5A1', fontSize: 11 }, icon: 'roundRect', itemWidth: 10, itemHeight: 3,
    },
    xAxis: {
      type: 'time', ...AXIS_STYLE, splitLine: { show: false },
      axisLabel: {
        color: '#9BA5A1', fontSize: 11,
        formatter: (value) => new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    },
    yAxis: [
      { type: 'value', min: 0, max: 100, name: 'Usage %', nameTextStyle: { color: '#9BA5A1', fontSize: 10 }, ...AXIS_STYLE },
      { type: 'value', name: 'ms', nameTextStyle: { color: '#9BA5A1', fontSize: 10 }, ...AXIS_STYLE, splitLine: { show: false } },
    ],
    series: [
      baseLine('CPU', '#22B8C3', pick('cpu_pct', 1)),
      baseLine('RAM', '#7C70E8', pick('ram_pct', 1)),
      baseLine('Swap', '#D99827', pick('swap_pct', 1)),
      baseLine('Disk latency', '#DF7B27', pick('disk_latency_ms', 2), 1),
    ],
  }, { notMerge: true, lazyUpdate: false });
}

/* ═══════════ critical events ═══════════ */

function renderAlerts(payload) {
  const alerts = (payload && payload.alerts) || [];
  const container = $('alerts');
  const badge = $('eventsBadge');

  badge.hidden = alerts.length === 0;
  badge.textContent = String(alerts.length);

  if (alerts.length === 0) {
    container.innerHTML = '<p class="empty">No alerts in this session.</p>';
    return;
  }

  container.innerHTML = alerts.map((alert, index) => `
    <div class="event-row">
      <span class="event-dot"></span>
      <div class="min-w-0">
        <div class="event-meta">
          <span class="event-time">${localTime(alert.timestamp_utc)}</span>
          <span class="event-severity">${escapeHtml(alert.status ?? 'Alert')}</span>
        </div>
        <p class="event-title">Slowdown risk detected — score ${fmt(alert.risk_score, 0)} / 100</p>
        <p class="event-body">
          CPU ${fmt(alert.cpu_pct, 1)}% · RAM ${fmt(alert.ram_pct, 1)}% · ${escapeHtml(alert.alert_reason ?? '')}
        </p>
        <p class="event-detail mono" id="event-detail-${index}" hidden>
          Session ${escapeHtml(alert.run_id ?? '—')}<br>
          Machine ${escapeHtml(alert.machine_id ?? '—')}<br>
          Prediction ${fmt(alert.prediction_id, 0)} · recorded ${escapeHtml(alert.timestamp_utc ?? '—')}
        </p>
      </div>
      <button type="button" class="btn btn-ghost" data-detail="${index}">Details</button>
    </div>`).join('');
}

/* Details is a client-side disclosure of the record already returned by
   /api/alerts — it triggers no request and claims no action on the alert. */
$('alerts').addEventListener('click', (event) => {
  const button = event.target.closest('[data-detail]');
  if (!button) return;
  const detail = $(`event-detail-${button.dataset.detail}`);
  if (!detail) return;
  detail.hidden = !detail.hidden;
  button.textContent = detail.hidden ? 'Details' : 'Hide';
});

/* ═══════════ model health (loaded once) ═══════════ */

function renderModel(model) {
  const metadata = model.metadata || {};
  const dataset = metadata.dataset || {};
  const artifacts = metadata.artifacts || {};
  const test = model.test_metrics || {};

  const facts = [
    ['Model', metadata.model_name ?? '—'],
    ['Task', metadata.task ?? '—'],
    ['Artifact', artifacts.model_path ?? '—'],
    ['Preprocessor', artifacts.tree_preprocessor_path ?? '—'],
    ['Input features', `${fmt(dataset.retained_original_features, 0)} → ${fmt(dataset.transformed_features, 0)} transformed`],
    ['Training rows', fmt(dataset.valid_modeling_rows, 0)],
    ['Machines · runs', `${fmt(dataset.machines, 0)} · ${fmt(dataset.modeling_runs, 0)}`],
    ['Decision threshold', fmt(metadata.decision_threshold, 2)],
    ['Frozen on', localDate(metadata.created_at_utc)],
  ];
  $('modelFacts').innerHTML = facts.map(([term, value]) => `
    <div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('');

  const tiles = [
    ['PR-AUC', fmt(test.pr_auc, 3)],
    ['ROC-AUC', fmt(test.roc_auc, 3)],
    ['Precision', fmt(test.precision, 3)],
    ['Recall', fmt(test.recall, 3)],
    ['F1', fmt(test.f1, 3)],
  ];
  $('testMetrics').innerHTML = tiles.map(([label, value]) => `
    <div class="tile">
      <div class="eyebrow">${label}</div>
      <div class="tile-value">${value}</div>
      <div class="tile-bar is-unscaled"></div>
    </div>`).join('');

  // Rendered in the order the evaluation produced. Never sorted, never filtered.
  $('byMachine').innerHTML = (model.by_machine || []).map((row) => {
    const weak = row.pr_auc !== null && row.pr_auc < 0.5
      ? '<span class="weak-tag">low performance</span>' : '';
    return `
      <tr>
        <td><span class="mono" title="${escapeHtml(row.machine_id)}">${escapeHtml(shortId(row.machine_id, 16))}</span>${weak}</td>
        <td>${fmt(row.rows, 0)}</td>
        <td>${fmt(row.runs, 0)}</td>
        <td>${fmtPercent(row.positive_rate)}</td>
        <td>${fmt(row.pr_auc, 3)}</td>
        <td>${fmt(row.roc_auc, 3)}</td>
        <td>${fmt(row.precision, 3)}</td>
        <td>${fmt(row.recall, 3)}</td>
        <td>${fmt(row.f1, 3)}</td>
      </tr>`;
  }).join('');

  // Honesty panel: wording comes verbatim from models/final_v1/model_metadata.json.
  $('limitations').innerHTML = (metadata.known_limitations || [])
    .map((line) => `<li>${escapeHtml(line)}</li>`).join('');

  const contract = metadata.score_contract || {};
  const notes = [
    ['Score contract', `${contract.risk_score ?? '—'} — ${contract.display_wording ?? ''}`],
    ['Calibration', contract.calibration_warning ?? '—'],
    ['Decision threshold', `${metadata.decision_threshold ?? '—'} — ${metadata.threshold_status ?? ''}`],
    ['Internal model status', metadata.model_role ?? '—'],
  ];
  $('contractNotes').innerHTML = notes.map(([term, description]) => `
    <div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(description)}</dd></div>`).join('');
}

function renderHealth(health) {
  state.mode = health.mode || 'live';
  const replay = state.mode === 'replay' ? health.replay : null;

  $('replayBadge').hidden = !replay;
  $('replaySource').hidden = !replay;
  if (replay) {
    $('replaySource').textContent =
      `Recorded episode replayed through the live pipeline — source run ${shortId(replay.source_run_id, 13)}`
      + ` · ${fmt(replay.rows, 0)} samples at ${fmt(replay.interval_seconds, 0)}s`;
    $('replaySource').title = replay.source_run_id;
  }

  $('healthLine').textContent =
    `mode ${state.mode} · model ${health.model_ready ? 'ready' : 'unavailable'} · database ${health.database_reachable ? 'reachable' : 'unreachable'} · api ${health.api_version}`;
}

/* ═══════════ polling ═══════════ */

async function pollLive() {
  if (state.liveSkipTicks > 0) { state.liveSkipTicks -= 1; return; }
  try {
    renderLive(await getJSON('/api/live'));
    registerSuccess();
  } catch (error) {
    registerFailure(error);   // last known values stay on screen
  }
}

async function pollHistory() {
  if (state.historySkipTicks > 0) { state.historySkipTicks -= 1; return; }
  try {
    const [history, alerts] = await Promise.all([
      getJSON('/api/history?minutes=10'),
      getJSON('/api/alerts?limit=12'),
    ]);
    renderCharts(history);
    renderAlerts(alerts);
    registerSuccess();
  } catch (error) {
    registerFailure(error);
  }
}

function startPolling() {
  if (state.liveTimer === null) state.liveTimer = setInterval(pollLive, LIVE_INTERVAL_MS);
  if (state.historyTimer === null) state.historyTimer = setInterval(pollHistory, HISTORY_INTERVAL_MS);
}

function stopPolling() {
  clearInterval(state.liveTimer);
  clearInterval(state.historyTimer);
  state.liveTimer = null;
  state.historyTimer = null;
}

/* Stop hitting the API while the tab is in the background. */
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopPolling();
  } else {
    state.liveSkipTicks = 0;
    state.historySkipTicks = 0;
    startPolling();
    pollLive();
    pollHistory();
  }
});

/* ═══════════ collector control ═══════════ */

$('collectorBtn').addEventListener('click', async () => {
  const collector = state.lastCollector;
  if (!collector || state.collectorBusy) return;

  state.collectorBusy = true;
  $('collectorBtn').disabled = true;
  try {
    await postJSON(collector.running ? '/api/collector/stop' : '/api/collector/start');
    hideNotice();
  } catch (error) {
    // 409 = already running, or a collector this dashboard does not own.
    showNotice(error.message, true);
  } finally {
    state.collectorBusy = false;
    pollLive();
  }
});

/* ═══════════ boot ═══════════ */

async function boot() {
  initCharts();
  showPage('overview');
  renderSample(null, null);
  renderAlerts(null);

  try {
    renderModel(await getJSON('/api/model'));
  } catch (error) {
    $('limitations').innerHTML = `<li>Model metadata could not be loaded — ${escapeHtml(error.message)}</li>`;
  }
  try {
    renderHealth(await getJSON('/api/health'));
  } catch (_) { /* footer detail only */ }

  await pollLive();
  await pollHistory();
  startPolling();
}

boot();
