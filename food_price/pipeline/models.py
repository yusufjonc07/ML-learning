"""The model zoo.

LinearRegression and RandomForest always exist (scikit-learn). XGBoost, LightGBM
and CatBoost are optional - each is imported lazily and simply omitted if its
package is not installed, so the pipeline runs with whatever is available and
picks up the gradient-boosters once you ``poetry install`` / ``pip install`` them.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def get_models() -> dict:
    """Return {name: unfitted estimator} for every model that can be constructed."""
    rs = config.RANDOM_STATE
    models: dict = {
        # Scaled so the linear baseline is well-conditioned across the ~80 features
        # (many on very different scales); trees below are scale-invariant.
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=rs
        ),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=rs, n_jobs=-1,
        )
    except Exception:  # noqa: BLE001
        print("[models] xgboost not installed - skipping")

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, random_state=rs, n_jobs=-1, verbose=-1,
        )
    except Exception:  # noqa: BLE001
        print("[models] lightgbm not installed - skipping")

    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=800, learning_rate=0.05, depth=8, random_seed=rs, verbose=0,
        )
    except Exception:  # noqa: BLE001
        print("[models] catboost not installed - skipping")

    return models
