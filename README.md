# Stock-Treadliner

**Founders**: Jasper Liu, Ngoc Vo, Ngoc Tu, Thuy Linh Pham, Nathanael Garcia, Nicholas Magtangob

**Description**: Given the stock dataset with daily OHLCV (Open, High, Low, Close, and Volume) values, we will use a machine learning pipeline to predict multiple stocks’ next-day logarithmic returns, defined as $y_{t+1} = \dfrac{log(close_{t+1})} {close_{t}}$, for the upcoming year.

## Get Started

Read this [final report document](https://docs.google.com/document/d/17oQ2WO7vD-_IzOdfuRfRw-2J3N8jhtK_BCtTDEFVWdg/edit?usp=sharing) to get a better understanding of the project

You can run the example notebooks directly from the `/experiment` directory to test the project workflows. However, for the scripts inside `/backend`, make sure to run them from the project root, e.g.:
```bash
python ./backend/evaluation.py
```

## Acknowledgement

This project uses:

1. [yfinance Stock Market Dataset](https://pypi.org/project/yfinance/) (© Ran Aroussi, licensed under the Apache License 2.0)

## Project Structure

1. `/experiment`: where the jupyter notebook and testing files live
2. `/backend`: where the machine learning pipeline and backend server live
3. `/frontend`: where the user interface lives
4. `/data`: where the processed and feature datasets live
5. `/model`: where meta-information about fine-tuned and ensembled models lives
6. `/graph`: where evaluation plots for each model live

