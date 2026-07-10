"""End-to-end training entry point.

    python train.py

Builds the panel, engineers features, does a time-based split, trains and
compares every available model, and writes metrics + plots to reports/.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # make config/pipeline/utils importable
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from pipeline.data import build_panel
from pipeline.features import assemble
from pipeline.split import time_split
from pipeline.models import get_models
from pipeline.evaluate import compare_models
from pipeline.explain import feature_importance, permutation_feature_importance, shap_summary


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, "\n1) BUILD PANEL\n" + "=" * 70)
    panel = build_panel()
    print(f"panel shape: {panel.shape}")

    print("=" * 70, "\n2) FEATURES\n" + "=" * 70)
    X, y, dates = assemble(panel)
    print(f"modelling matrix: {X.shape[0]} rows x {X.shape[1]} features "
          f"({dates.min().date()} -> {dates.max().date()})")

    print("=" * 70, "\n3) TIME SPLIT\n" + "=" * 70)
    X_train, X_test, y_train, y_test = time_split(X, y, dates)
    dates_test = dates[dates >= pd.Timestamp(config.SPLIT_DATE)].reset_index(drop=True)
    print(f"train: {len(X_train):>6} rows  (< {config.SPLIT_DATE})")
    print(f"test : {len(X_test):>6} rows  (>= {config.SPLIT_DATE})")
    if len(X_test) == 0 or len(X_train) == 0:
        raise SystemExit("Empty train or test set - adjust config.SPLIT_DATE.")

    print("=" * 70, "\n4) TRAIN & COMPARE\n" + "=" * 70)
    models = get_models()
    print("models:", ", ".join(models))
    table, preds, fitted = compare_models(models, X_train, y_train, X_test, y_test)

    print("=" * 70, "\n5) RESULTS (sorted by RMSE)\n" + "=" * 70)
    print(table.round(3).to_string())
    table.round(4).to_csv(config.REPORTS_DIR / "metrics.csv")
    best = table.index[0]
    print(f"\nbest model: {best}")

    # Feature importance for the best model. Permutation importance (on the test
    # set) is the headline - it's model-agnostic and scale-invariant; model-native
    # importance is saved as a secondary view.
    print("\ncomputing permutation importance on the test set...")
    perm = permutation_feature_importance(fitted[best], X_test, y_test,
                                          random_state=config.RANDOM_STATE)
    perm.to_csv(config.REPORTS_DIR / "feature_importance.csv", header=["perm_importance_rmse"])
    ax = perm.head(20).iloc[::-1].plot.barh(
        figsize=(8, 7), title=f"{best} - permutation importance (RMSE increase), top 20")
    ax.set_xlabel("RMSE increase when shuffled")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "feature_importance.png", dpi=120)
    plt.close("all")
    print("top 10 features (permutation importance):")
    print(perm.head(10).round(3).to_string())

    native = feature_importance(fitted[best], X.columns)
    if native is not None:
        native.to_csv(config.REPORTS_DIR / "feature_importance_native.csv", header=["importance"])

    # actual vs predicted, monthly mean over the test window
    comp = pd.DataFrame({"date": dates_test.values, "actual": y_test.values, "pred": preds[best]})
    monthly = comp.groupby("date")[["actual", "pred"]].mean()
    monthly.plot(figsize=(9, 5), marker="o", title=f"{best}: monthly-mean price, test window")
    plt.ylabel("price (UZS)")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "predictions.png", dpi=120)
    plt.close("all")

    # SHAP for the best model if it's a tree and shap is installed
    shap_path = shap_summary(fitted[best], X_test, config.REPORTS_DIR / "shap_summary.png")

    print("\nreports written to:", config.REPORTS_DIR)
    for f in ["metrics.csv", "feature_importance.png", "predictions.png"] + (
        ["shap_summary.png"] if shap_path else []
    ):
        print("  -", f)


if __name__ == "__main__":
    main()
