"""Model explainability: feature importance and (optional) SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd


def permutation_feature_importance(model, X, y, n_repeats: int = 5, random_state: int = 0) -> pd.Series:
    """Model-agnostic importance = RMSE increase when each feature is shuffled.

    Computed on the (held-out) data you pass in, so it measures real predictive
    contribution - not a scale artifact like |coef_| or the split-count bias of
    tree impurity. This is the ranking to trust; read it as clusters, since
    collinear features share credit.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=list(X.columns)).sort_values(ascending=False)


def feature_importance(model, feature_names) -> pd.Series | None:
    """Model-native importance (feature_importances_ or |coef_|), or None.

    Kept as a secondary view; prefer `permutation_feature_importance`. Returns
    None for pipelines/models that expose neither attribute.
    """
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.ravel(model.coef_)).astype(float)
    else:
        return None
    return pd.Series(imp, index=list(feature_names)).sort_values(ascending=False)


def shap_summary(model, X, path, max_display: int = 20):
    """Save a SHAP summary plot for a tree model, or return None if unavailable.

    Uses a sample of X for speed. Silently no-ops when shap isn't installed or the
    model isn't tree-based (keeps the pipeline runnable everywhere).
    """
    try:
        import shap
    except Exception:
        print("[shap] not installed - skipping SHAP plot")
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample = X.sample(min(len(X), 2000), random_state=0) if len(X) > 2000 else X
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        shap.summary_plot(values, sample, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close("all")
        return path
    except Exception as exc:  # noqa: BLE001 - explainability is best-effort
        print(f"[shap] skipped ({exc})")
        return None
