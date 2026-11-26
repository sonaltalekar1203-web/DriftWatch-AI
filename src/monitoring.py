"""End-to-end DriftWatch AI demo: train, simulate drift, monitor, and report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from .drift_detection import DriftConfig, detect_feature_drift, detect_dataset_drift

ROOT = Path(__file__).resolve().parents[1]


def run():
    rng = np.random.default_rng(42)
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_ref, X_test, y_ref, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=250, random_state=42,
                                                 class_weight="balanced"))
    model.fit(X_ref, y_ref)
    baseline = {"accuracy": accuracy_score(y_test, model.predict(X_test)),
                "f1": f1_score(y_test, model.predict(X_test)),
                "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}

    pool = X_test.copy()
    scenarios = {
        "normal": pool + rng.normal(0, 0.02, pool.shape) * X_ref.std().to_numpy(),
        "moderate": pool.copy(),
        "significant": pool.copy()
    }
    # Shift a subset of numeric features to simulate changing measurement conditions.
    for col in list(X.columns)[:6]:
        scenarios["moderate"][col] += 0.6 * X_ref[col].std()
    for col in list(X.columns)[:12]:
        scenarios["significant"][col] += 1.25 * X_ref[col].std()

    results = {}
    for name, batch in scenarios.items():
        feature_report = detect_feature_drift(X_ref, batch, DriftConfig())
        dataset_report = detect_dataset_drift(feature_report, DriftConfig())
        pred = model.predict(batch)
        prob = model.predict_proba(batch)[:, 1]
        metrics = {"accuracy": accuracy_score(y_test, pred),
                   "f1": f1_score(y_test, pred),
                   "roc_auc": roc_auc_score(y_test, prob)}
        degraded = (baseline["accuracy"] - metrics["accuracy"] > 0.05 or
                    baseline["f1"] - metrics["f1"] > 0.05)
        status = "CRITICAL" if degraded else (
            "WARNING" if dataset_report["dataset_drift"] else "HEALTHY")
        results[name] = {"status": status, "metrics": metrics,
                         "drift": dataset_report,
                         "features": feature_report.to_dict(orient="records")}

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    payload = {"baseline": baseline, "scenarios": results}
    (ROOT / "results" / "monitoring_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for name, item in results.items():
        rows.append({"scenario": name, "status": item["status"],
                     "drifted_features": item["drift"]["n_drifted_features"],
                     **item["metrics"]})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    table.set_index("scenario")[["accuracy", "f1", "roc_auc"]].plot(
        kind="bar", ax=ax, ylim=(0, 1), title="Model performance by scenario")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "scenario_metrics.png", dpi=150)
    html = table.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    (ROOT / "reports" / "monitoring_report.html").write_text(
        "<html><head><title>DriftWatch AI</title></head><body>"
        "<h1>DriftWatch AI Monitoring Report</h1>"
        "<p>Simulated production data drift and model performance.</p>"
        + html + "</body></html>", encoding="utf-8")
    print(table.to_string(index=False))
    return payload


if __name__ == "__main__":
    run()
