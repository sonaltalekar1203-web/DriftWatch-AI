# DriftWatch AI

**ML Model Monitoring & Data Drift Detection**

DriftWatch AI is a Python-based MLOps demonstration that detects changes between reference and simulated production data, monitors prediction distributions, and evaluates model performance.

## Features
- Population Stability Index (PSI)
- Two-sample Kolmogorov–Smirnov (KS) test
- Benjamini–Hochberg correction for multiple comparisons
- Feature-level and dataset-level drift reports
- Simulated normal, moderate, and significant drift
- Random Forest model with performance metrics
- HTML monitoring report and JSON results

## Monitoring dashboard

[Open the interactive model monitoring report](reports/monitoring_report.html). The report compares four production scenarios and includes PSI/KS analysis, model-performance metrics, feature-level results, and recommendations.

## Quick start
```bash
pip install -r requirements.txt
python -m src.monitoring
```
Open `reports/monitoring_report.html` after the run.

## Tests
```bash
pip install pytest
pytest -q
```

## Project structure
```
DriftWatch-AI/
├── src/
│   ├── drift_detection.py
│   └── monitoring.py
├── tests/
│   └── test_drift_detection.py
├── reports/
│   └── monitoring_report.html
├── results/                 # generated metrics
├── requirements.txt
└── README.md
```

## Tech stack
Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib.

## Dataset and limitations
Uses the Breast Cancer Wisconsin (Diagnostic) dataset bundled with scikit-learn. Drift scenarios are simulated for demonstration; this is not a clinically validated medical monitoring tool. Model metrics are dataset- and split-dependent and should not be interpreted as clinical performance.
