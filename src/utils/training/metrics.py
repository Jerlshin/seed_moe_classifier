from __future__ import annotations

from typing import Any

import numpy as np


def classification_report_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except Exception:
        return {}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def classification_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    positive_label: int = 1,
) -> dict[str, Any]:
    try:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve, roc_curve
    except Exception:
        return {}

    if y_score.ndim > 1:
        score = y_score[:, positive_label]
    else:
        score = y_score

    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true, score, pos_label=positive_label)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        y_true,
        score,
        pos_label=positive_label,
    )
    calib_prob_true, calib_prob_pred = calibration_curve(y_true, score, n_bins=10)

    return {
        "roc": {
            "fpr": roc_fpr,
            "tpr": roc_tpr,
            "thresholds": roc_thresholds,
        },
        "precision_recall": {
            "precision": pr_precision,
            "recall": pr_recall,
            "thresholds": pr_thresholds,
        },
        "calibration": {
            "prob_true": calib_prob_true,
            "prob_pred": calib_prob_pred,
        },
    }
