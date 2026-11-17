# Weather Station Data Quality Anomaly Detector

A local Python/Streamlit application for non-destructive retrospective weather telemetry auditing. It combines a 240-tree Random Forest, 17 contextual features, and physical guardrails, with four dashboard tabs covering auditing, fault experiments, model evidence, and presentation material.

## Run

Python 3.11–3.13 is recommended; development was verified with Python 3.13.

```sh
cd weather_anomaly_detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the localhost address printed by Streamlit. First startup automatically generates data and trains the model if the artifacts are absent. No API keys or external data services are required. The included CSV is **synthetic**, not downloaded station observations.

## Input and output

CSV columns: `date` (YYYY-MM-DD), `station_id`, `tmax` and `tmin` (°C), `rainfall` (mm/day). Empty measurement cells are allowed and flagged. Invalid dates, duplicate station/date pairs, nonnumeric measurements, and infinities produce validation errors. Additional columns are preserved except reserved output names, which are replaced by audit results. `scenario_id` is reserved for internal training experiments and rejected on upload.

Original measurement values, including nulls, are preserved. Export appends probability, rule flags/reasons, hybrid alerts and diagnostic clues. Display filtering retains the full input as context for neighboring stations and days. Missing entire dates are not fabricated: gaps break adjacency and flatline runs. Rainfall-only anomalies are shown on the temperature chart at the available temperature, or zero if both temperatures are absent.

## Architecture

- `src/data_generator.py`: correlated 2018–2023 daily data for four New York stations, shared heatwaves/cold snaps and six synthetic fault modes.
- `src/features.py`: exactly 17 features, scenario isolation, calendar rolling windows and safe single-station/all-null fallbacks.
- `src/rules.py`: missing values, temperature ordering, negative rainfall and broad physical bounds.
- `src/model.py`: 240 trees, 2018–2021 training, 2022 RF F1 threshold calibration, 2023 test inference and saved artifact.
- `src/evaluator.py`: hybrid precision/recall/F1, RF ROC-AUC, confusion matrix and per-fault recall.
- `app.py`: dashboard and automatic bootstrap.

Clean and corrupted copies are isolated using `scenario_id`. Features are computed independently for each temporal partition to avoid boundary leakage; climatological expectations use a fixed NY seasonal curve. Neighbor medians exclude the target station. Rolling imputation affects feature computation only. The forward-day feature deliberately makes this a historical batch system. Retrain after changing feature/model code with `python -m src.model` and restart Streamlit to refresh its cache. Only load trusted, locally generated joblib artifacts.

## Verification

```sh
python -c "from src.data_generator import ensure_dataset; from src.model import train_and_save_model; ensure_dataset(); train_and_save_model()"
python -m unittest discover -s tests -v
streamlit run app.py --server.headless true
```

The Model Evidence tab reports actual held-out results at the selected threshold, including clean-row false positives. A lower threshold generally increases recall and false positives. Rule and ML alert counts overlap.

## Viva and limitations

The Learn & Present tab contains ten defense questions and an interactive heatwave/spike comparison. Context matters: shared extreme weather is different from an isolated jump or repeated sensor value. Rules cover known invalid values while the forest learns combinations of temporal and spatial evidence. Diagnostics rank deviations from clean training statistics weighted by global feature importance; they are heuristics, not causal feature attributions.

Synthetic performance does not establish field accuracy. Small early drift and the first day of a stuck sensor can be indistinguishable from clean readings. Single-station audits lose spatial evidence. Regional correlated sensor errors can evade peer checks. The NY seasonal prior may not transfer to other climates. The detector flags records for human review, does not prove corruption, and never repairs or deletes measurements. Live IoT ingestion, forecasts and public weather warnings are outside scope.

## Upload testing sample

`data/ny_anomaly_upload_test.csv` contains 1,460 daily records (four stations throughout 2023), generated with a separate random seed. Upload it under **Station Overview → Custom CSV Upload**. It contains 34 labeled fault-window rows, all at **NY_CENTRAL_PARK**: February 10 spike; March 10–16 flatline; April 10–24 drift; May 10–11, 15–16 and 20–21 missing fields; June 10 inverted temperatures; June 20 negative rain; August 10–12 neighbor disagreement. The July regional heatwave remains intact. The first flatline day is part of the injected event but is numerically unchanged.

`data/ny_anomaly_answer_key.csv` provides row-level ground truth and original versus injected measurements. Keep it separate when uploading the five-column sample. Both files are also downloadable from the Custom CSV Upload section. Detection is imperfect; these files are for exercising the application, not an independent real-world benchmark.
