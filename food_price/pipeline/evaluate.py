"""Regression metrics and a model-comparison table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R2, MAPE(%). MAPE ignores rows where the true price is 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE": float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100),
    }


def compare_models(models: dict, X_train, y_train, X_test, y_test):
    """Fit every model, score it on the test set, and return a sorted table.

    Returns (table, preds, fitted): the metrics DataFrame (best RMSE first), the
    per-model test predictions, and the fitted estimators.
    """
    rows, preds, fitted = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows[name] = regression_metrics(y_test, pred)
        preds[name] = pred
        fitted[name] = model
        print(f"[eval] {name:16s} "
              + "  ".join(f"{k}={v:,.3f}" for k, v in rows[name].items()))
    table = pd.DataFrame(rows).T.sort_values("RMSE")
    return table, preds, fitted
