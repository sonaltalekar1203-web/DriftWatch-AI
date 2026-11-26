"""Statistical data-drift detection utilities for DriftWatch AI."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DriftConfig:
    bins: int = 10
    psi_threshold: float = 0.20
    ks_alpha: float = 0.05
    dataset_drift_share: float = 0.30
    epsilon: float = 1e-4


def population_stability_index(reference, current, bins=10, epsilon=1e-4):
    """Calculate PSI using quantile bins defined by the reference sample."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if not len(ref) or not len(cur):
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref)
    b = np.histogram(cur, edges)[0] / len(cur)
    a, b = np.clip(a, epsilon, None), np.clip(b, epsilon, None)
    return float(np.sum((b - a) * np.log(b / a)))


def _bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def detect_feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                         config: DriftConfig | None = None) -> pd.DataFrame:
    """Return PSI, KS statistics, adjusted p-values and drift verdict per feature."""
    cfg = config or DriftConfig()
    rows = []
    for col in reference.columns:
        ref = reference[col].dropna().to_numpy()
        cur = current[col].dropna().to_numpy()
        psi = population_stability_index(ref, cur, cfg.bins, cfg.epsilon)
        ks = stats.ks_2samp(ref, cur)
        rows.append({"feature": col, "psi": psi, "ks_statistic": float(ks.statistic),
                     "ks_p_value": float(ks.pvalue)})
    report = pd.DataFrame(rows)
    report["ks_p_adjusted"] = _bh_adjust(report["ks_p_value"])
    report["drift_detected"] = ((report["psi"] >= cfg.psi_threshold) |
                                (report["ks_p_adjusted"] < cfg.ks_alpha))
    return report.sort_values("psi", ascending=False).reset_index(drop=True)


def detect_dataset_drift(feature_report: pd.DataFrame,
                         config: DriftConfig | None = None) -> dict:
    cfg = config or DriftConfig()
    count = int(feature_report["drift_detected"].sum())
    total = len(feature_report)
    share = count / total if total else 0.0
    return {"n_features": total, "n_drifted_features": count,
            "drift_share": round(share, 4),
            "dataset_drift": bool(share >= cfg.dataset_drift_share),
            "drifted_features": feature_report.loc[
                feature_report["drift_detected"], "feature"].tolist()}
