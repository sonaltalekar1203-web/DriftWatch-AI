import numpy as np
import pandas as pd
from src.drift_detection import (DriftConfig, population_stability_index,
                                 detect_feature_drift, detect_dataset_drift)


def test_identical_samples_have_near_zero_psi():
    x = np.linspace(0, 1, 500)
    assert population_stability_index(x, x) < 1e-10


def test_shifted_samples_have_higher_psi():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(2, 1, 1000)
    assert population_stability_index(ref, cur) > 0.2


def test_feature_and_dataset_drift():
    rng = np.random.default_rng(7)
    ref = pd.DataFrame({"stable": rng.normal(0, 1, 500),
                        "shifted": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"stable": rng.normal(0, 1, 500),
                        "shifted": rng.normal(2, 1, 500)})
    cfg = DriftConfig(dataset_drift_share=0.5)
    feature = detect_feature_drift(ref, cur, cfg)
    dataset = detect_dataset_drift(feature, cfg)
    assert feature.loc[feature.feature == "shifted", "drift_detected"].iloc[0]
    assert dataset["dataset_drift"]
