# XGBoost Multi-Target Risk Prediction Models

This repository houses two completely decoupled, self-contained health risk prediction systems built with calibrated XGBoost classifiers.

---

## Repository Structure

The project is structured into two main independent folders:

### 1. [CurrentModel/](file:///c:/Users/KISHORE%20SHYAM.%20V/PycharmProjects/PythonProject4/CurrentModel)
Predicts a patient's **current critical risk status** (binary target `CURRENT_RISK` built from raw labels being "High" or "Very High").
* **`data/`**: Datasets containing training cohort and raw risk scores.
* **`src/`**: Model training (`train_current_model.py`) and isotonic probability calibration (`calibrate_current_model.py`).
* **`utils/`**: Feature ablation experiments and accuracy scoring.
* **`outputs/`**: Model binaries, imputers, calibrators, and predictions.

### 2. [FutureModel/](file:///c:/Users/KISHORE%20SHYAM.%20V/PycharmProjects/PythonProject4/FutureModel)
Predicts three distinct **future outcomes** for the year following the snapshot date (High Utilization, Clinical Deterioration, and Healthcare Escalation).
* **`data/`**: Raw original patient features, model-ready leakage-free datasets, and synthetically drifted populations.
* **`src/`**:
  * `build_dataset.py`: Decouples target logic to construct leakage-free features.
  * `train_future_models.py` / `calibrate_future_models.py`: Model fitting and calibration.
  * `risk_engine.py`: Unified engine that integrates current risk forecasts (if available) with future outcome probabilities to profile patients.
  * `patient_monitor.py`: Scans clinical cohorts for outliers, temporal risk spikes, and critical alarms.
* **`utils/`**: Data drift analyses, target check validation, and temporal audits.
* **`outputs/`**: Future model binaries, calibration parameters, combined predictions, and alert files.

---

## Getting Started

### To run the Current Risk Model pipeline:
```bash
python CurrentModel/src/train_current_model.py
python CurrentModel/src/calibrate_current_model.py
python CurrentModel/utils/show_accuracy.py
```

### To run the Future Risk Model pipeline:
```bash
python FutureModel/src/build_dataset.py
python FutureModel/src/train_future_models.py
python FutureModel/src/calibrate_future_models.py
python FutureModel/src/risk_engine.py
python FutureModel/src/patient_monitor.py
```
*(Note: `FutureModel/src/risk_engine.py` will automatically detect and integrate current risk predictions if they have been generated in the `CurrentModel/outputs/` directory).*
