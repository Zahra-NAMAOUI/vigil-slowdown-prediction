# Vigil — Predictive slowdown monitoring

Vigil predicts whether a computer will slow down in the next five
minutes, from live system metrics.

It covers the full chain: a collection agent, a labeling rule, a
time-series feature pipeline, a trained model, and a real-time
dashboard that serves predictions from frozen artifacts.

---

## The problem

Nothing in raw system metrics tells you a slowdown is coming. The
target had to be built.

`slowdown_now` comes from Rule C: a slowdown is present when at least
one severe pressure signal, or two simultaneous moderate ones, appear
across CPU, RAM, swap, disk latency and context switches.

`slowdown_in_5min` is the predictive target: 1 if `slowdown_now`
occurs within the next five minutes, inside the same continuous
segment. Observations without a full five-minute future horizon are
excluded.

---

## Data

| | |
|---|---|
| Raw observations | 153,521 |
| With a valid 5-minute horizon | 133,016 |
| Machines | 4 |
| Collection runs | 67 |
| Period | 23 July → 12 August 2026 |
| Positive rate | 37.9% |

Sampling every 2 seconds. Time gaps are detected and split into
continuous segments; no rolling window ever crosses a gap.

---

## Pipeline
collection → cleaning → temporal analysis → segmentation
→ gap validation → Rule C labeling → 5-minute target
→ feature engineering → chronological split → preprocessing
→ model comparison → robustness → distribution shift
→ final test evaluation → frozen V1

**Features.** 20 raw metrics → 243 candidates. Rolling windows
(30s / 60s / 120s) with mean, max, min, std, change and missing rate,
plus intra-segment differences and missingness indicators. All windows
are time-based and strictly backward-looking, computed within
`machine_id + run_id + segment_id`. After preprocessing: 360 features.

**Split.** Chronological, by machine, whole runs only. No run and no
row is shared between train, validation and test. The most recent runs
form the test set, which was evaluated exactly once.

| | Rows | Runs | Positive rate |
|---|---|---|---|
| Train | 82,860 | 39 | 34.0% |
| Validation | 22,212 | 7 | 49.1% |
| Test | 27,944 | 9 | 40.6% |

---

## Model

LightGBM, selected over logistic regression, random forest and
histogram gradient boosting. Hyperparameter search did not improve on
the baseline configuration, which was therefore kept.

Final evaluation, on a test set never used for any decision:

| Metric | Test |
|---|---|
| PR-AUC | 0.9519 |
| ROC-AUC | 0.9549 |
| Precision | 0.8154 |
| Recall | 0.9221 |
| F1 | 0.8655 |

Recall is prioritised: a missed slowdown costs more than a false alert.

**Leakage audit.** Verified that the target is absent from the
inputs, that no future value enters any feature, that no centred
window is used, and that all temporal operations stay within one
segment. No leakage found.

---

## Known limitations

These are documented deliberately. V1 is an experimental prototype.

- The score is **not a calibrated probability**.
- The 0.50 threshold is a development reference, not validated for
  production.
- Performance varies sharply across machines. On one machine the model
  reaches PR-AUC ≈ 1.0; on another it collapses to 0.048 with recall
  near zero.
- Significant covariate shift exists between machines, with a probable
  concept drift: on the difficult run, only 17% of positive
  observations show a Rule C pressure at the current instant, against
  54% in training.
- Rule C was defined on data dominated by Windows machines. On macOS,
  where swap is retained long after the memory pressure that caused it,
  the swap criterion can stay satisfied almost permanently — which
  saturates the prediction. This is a finding about the target
  definition, not about model quality.

---

## Real-time dashboard

FastAPI backend + static front end (vanilla JS, ECharts). The
inference chain is frozen and self-verifying.

**Guarantees enforced at startup**

- SHA-256 of the model and the preprocessor are recomputed and compared
  against `models/final_v1/model_metadata.json`. On mismatch the
  application refuses to start rather than serving a different model.
- The order and count of features are checked strictly (228 → 360).
- Train/live parity is proven by `dashboard/diagnostic.py`: the 228
  features recomputed live match the training dataset with a maximum
  absolute error of 0.0.

**Warm-up.** 120 seconds of uninterrupted history are required before
the first prediction. Any gap over 10 seconds opens a new segment and
resets the counter. No score is displayed during warm-up.

---

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt

uvicorn dashboard.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and start the collector from the header.

Prove the inference chain:

```bash
python -m dashboard.diagnostic
```

---

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Cross-platform collection agent (psutil), merging, cleaning |
| `notebooks/` | 01 → 18, the full pipeline in order |
| `dashboard/` | FastAPI API, inference, live features, web front end |
| `models/` | Frozen V1 artifacts and preprocessors |
| `reports/` | Metrics, robustness and distribution shift analyses |
| `data/` | Not versioned (~2 GB) |

Notebook `16b` retrains and overwrites the frozen V1 artifacts. Do not
run it.

---

## Next steps

- Wire SHAP into the dashboard to explain each alert
- Convert notebooks 08 and 12 into reproducible scripts
- Collect data from more machines and operating systems to reduce shift
- Re-examine Rule C's swap criterion across operating systems
