# Stock-Treadliner

**Founders**: Jasper Liu, Ngoc Vo, Ngoc Tu, Thuy Linh Pham, Nathanael Garcia, Nicholas Magtangob

**Description**: Given the stock dataset with daily OHLCV (Open, High, Low, Close, and Volume) values, we will use a machine learning pipeline to predict multiple stocks’ next-day logarithmic returns, defined as $y_{t+1} = \dfrac{log(close_{t+1})} {close_{t}}$, for the upcoming year.

## References

1. [Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

### Data Sources

- **Stock Market Data**: Historical OHLCV (Open, High, Low, Close, Volume) data sourced from [Yahoo Finance](https://finance.yahoo.com/) via the [yfinance](https://pypi.org/project/yfinance/) Python library.
  - Library: `yfinance` 
  - Citation: Aroussi, R. (2024). yfinance: Download market data from Yahoo! Finance API. PyPI. https://pypi.org/project/yfinance/
  - Stocks Analyzed: AAPL, GOOGL, MSFT, AMZN, TSLA
  - Time Period: January 2020 - December 2024

## Project Structure
```
Stock-Treadliner/
├── backend/                 # FastAPI backend server
├── frontend/                # Next.js frontend application
├── experiment/              # Jupyter notebooks for experimentation
│   └── 01-preprocessing.ipynb    # Data preprocessing pipeline
├── data/                    # Processed datasets
│   └── processed/           # Cleaned and preprocessed stock data
│       ├── combined_stock_data.csv      # All stocks combined
│       ├── AAPL_cleaned.csv             # Individual stock files
│       ├── GOOGL_cleaned.csv
│       ├── MSFT_cleaned.csv
│       ├── AMZN_cleaned.csv
│       ├── TSLA_cleaned.csv
│       └── stock_visualization.png      # Data visualization
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## Project Workflow

to be filled
