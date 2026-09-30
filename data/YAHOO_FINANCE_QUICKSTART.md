# Financial XAI - Yahoo Finance Quick Start

This guide shows how to download real market data from Yahoo Finance and run the stock forecasting pipeline.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

Run the demo that downloads data and trains a model:

```bash
python -m financial_xai.demo_yahoo
```

This will:
1. Download 10 years of daily data for 5 stocks (AAPL, MSFT, AMZN, GOOGL, TSLA)
2. Engineer features (returns, volatility, momentum, RSI, MACD, volume, Bollinger bands)
3. Train a random forest model to predict 5-day forward returns
4. Print model evaluation metrics
5. Show the top 10 most important features using SHAP

## Manual usage

Download data for specific stocks:

```python
from financial_xai.yahoo_downloader import download_and_save

data = download_and_save(
    symbols=["AAPL", "MSFT", "AMZN"],
    output_path="data/my_stocks.csv",
    start="2015-01-01",
    end="2025-01-01",
)
```

Run the forecasting pipeline on your data:

```python
from financial_xai.real_data_pipeline import run_market_csv_pipeline

results = run_market_csv_pipeline(
    csv_path="data/my_stocks.csv",
    target_period=5,
    model_type="random_forest",
    return_explanation=True,
)

print(results["metrics"])
print(results["feature_importance"])
```

## Expected data format

The CSV should have columns:
- `date` — trading date
- `asset_id` — stock ticker symbol
- `open` — opening price
- `high` — highest price of the day
- `low` — lowest price of the day
- `close` — closing price
- `volume` — trading volume

Example:

```
date,asset_id,open,high,low,close,volume
2015-01-02,AAPL,111.39,111.44,107.35,109.33,53204626
2015-01-05,AAPL,108.29,108.65,107.86,108.59,64285491
...
```

## Customizing the pipeline

Edit `src/financial_xai/demo_yahoo.py` to:
- change the list of stocks
- adjust the start/end dates
- switch model type (logistic, gradient_boosting)
- change the forecast horizon (target_period parameter)

## Next steps

- Integrate your own market data source
- Add more features (correlation, relative strength, macro signals)
- Compare model types and tune hyperparameters
- Build a portfolio-level prediction summary
- Deploy as an API or dashboard
