# Tashkent Food-Price Forecasting

Monthly retail food-price forecasting for **Tashkent city** (12 districts +
city-wide), by product. A clean, reproducible ML pipeline: collect external
drivers → merge → engineer time-series features → time-based split → compare
tree models → explain.

## Structure

```
food_price/
├── config.py              # paths, keys, feature settings, split date, seeds
├── train.py               # ENTRY POINT: python train.py
│
├── datasets/              # data (raw + generated caches)
│   ├── prices.csv         # raw target: year, month, product, district, price
│   ├── cbu/               # raw CBU exchange-rate .xls
│   ├── weather_monthly.csv        # generated (utils/weather.py)
│   ├── population_monthly.csv      # generated (utils/population.py)
│   ├── fuel_monthly.csv            # cached  (utils/fuel.py, first run)
│   └── currency_monthly.csv        # cached  (utils/currency.py, first run)
│
├── utils/                 # external-data collectors (one concern each)
│   ├── curr_scrap.py / currency.py     # CBU USD/UZS -> monthly
│   ├── fuel_scrap.py  / fuel.py         # goldenpages fuel -> monthly
│   ├── weather.py                      # Open-Meteo -> national monthly weather
│   ├── population.py                   # annual -> monthly district population
│   └── featuring.py                    # lags / rolling / calendar features
│
├── pipeline/              # the ML pipeline (imported by train.py)
│   ├── data.py            # load + merge every source -> panel
│   ├── features.py        # panel -> numeric modelling matrix (X, y, dates)
│   ├── split.py           # time-based train/test split
│   ├── models.py          # model zoo (LR, RF, XGBoost, LightGBM, CatBoost)
│   ├── evaluate.py        # MAE / RMSE / R2 / MAPE + comparison table
│   └── explain.py         # feature importance + optional SHAP
│
├── reports/               # generated: metrics.csv, *.png, feature_importance.csv
└── models/                # (reserved for saved model artifacts)
```

## Run

```bash
cd food_price
python train.py
```

First run builds and caches the fuel/currency monthly CSVs (fuel needs network).
Weather and population CSVs are already generated; regenerate any source via its
`utils/` module. Missing optional libraries (xgboost/lightgbm/catboost/shap) are
skipped automatically — install them with `poetry install` (they're in
`pyproject.toml`) to enable the full comparison.

## Data sources & merge keys

| Source | Module | Grain | Merge key |
|---|---|---|---|
| Food prices (target) | opendata.tashkent.uz | product × district × month | — |
| Fuel (AI-80…Diesel) | goldenpages.uz | city × month | `date` |
| USD/UZS | cbu.uz | month | `date` |
| Weather | Open-Meteo (ERA5) | national × month | `date` |
| Population | stat.uz (annual→monthly) | district × month | `["district","date"]` |

## Features

- **Price history** (per product×district, no leakage): `price_lag_{1,2,3,12}`,
  `price_roll_{mean,std}_{3,6,12}`.
- **Calendar**: `month_of_year`, `quarter`, `month_sin`, `month_cos`.
- **Exogenous**: fuel grades, USD, weather block, population.
- **Categoricals**: `product_code`, `district_code`.

## Notes on honesty

- **Weather** is national (all Tashkent districts share one climate) → temporal,
  not spatial signal; a secondary, lagged driver, strongest for perishables.
- **Population** is interpolated from annual anchors (monthly figures don't
  exist) and the seed numbers are approximate — replace with official stat.uz data.
- Read tree importances as **clusters, not exact ranks** (features are collinear);
  prefer permutation importance / SHAP on the time-based test split.
```
