"""ML pipeline for monthly Tashkent food-price forecasting.

Stages (see the individual modules):
    data     -> load and merge every source into one clean monthly panel
    features -> lags, rolling stats, calendar encodings, categorical codes
    split    -> time-based train/test split (no shuffling, no leakage)
    models   -> the model zoo (LR, RF, XGBoost, LightGBM, CatBoost)
    evaluate -> MAE / RMSE / R2 / MAPE and a model-comparison table
    explain  -> feature importance and (optional) SHAP
Run everything with ``python train.py`` from the food_price/ directory.
"""
