# Financial XAI

A Python project for building explainable AI systems for financial forecasting and portfolio analysis.

## Goal

This repository focuses on a real stock and portfolio forecasting workflow using machine learning and explainable AI.

## What is included

- feature engineering for OHLCV market data
- portfolio-level synthetic data generation
- model training and validation
- SHAP-based explainability for feature impact
- a notebook and example data workflow for experimentation

## Real-data workflow

The project now includes a sample market dataset and a notebook that shows how to:

1. load historical market data from CSV
2. engineer signals like returns, momentum, volatility, RSI, and MACD
3. build a forecasting target
4. train a model
5. inspect the top drivers behind each prediction

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m financial_xai
```

## Web dashboard

Install the project and dependencies, then launch the dashboard from the repository root:

```bash
pip install -e .
streamlit run src/financial_xai/dashboard.py
```

The dashboard can train on generated sample data or an uploaded OHLCV CSV. CSV files must include `date`, `asset_id`, `open`, `high`, `low`, `close`, and `volume`, with at least 140 rows per asset. The interface reports time-ordered holdout metrics and global SHAP feature importance.

Or run the notebook workflow:

```bash
jupyter notebook notebooks/stock_forecasting_experiment.ipynb
```

## Project structure

```text
financial-xai/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── data/
│   ├── README.md
│   └── sample_market_data.csv
├── notebooks/
│   └── stock_forecasting_experiment.ipynb
├── src/
│   └── financial_xai/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── dataset.py
│       ├── explainability.py
│       ├── features.py
│       ├── main.py
│       ├── model.py
│       ├── pipeline.py
│       └── real_data_pipeline.py
├── tests/
│   └── test_pipeline.py
└── .venv/
```

## Recommended next research steps

- replace the sample CSV with real market data
- compare multiple models on real financial histories
- add rolling window features and event features
- analyze portfolio risk using explainable outputs
- expose predictions through an API or dashboard
