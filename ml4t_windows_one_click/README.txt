Windows local runner for ML4T Chapter 7 per-ticker RMSE.

This runner reuses the current folder:

  ml4t_windows_one_click\
    machine-learning-for-trading\
    ml4t_ch7_per_ticker_rmse.py
    run_ml4t_ch7_per_ticker_rmse.ps1

It does not clone the official repo, does not pull updates, and does not create
a virtual environment. It uses the existing Python environment from this
project, or py -3/python from PATH. If required Python packages are missing, it
installs them into that selected environment with pip.

Usage:
1. Open PowerShell as normal user.
2. cd to this folder:
   cd ml4t_windows_one_click
3. Run:
   powershell -ExecutionPolicy Bypass -File .\run_ml4t_ch7_per_ticker_rmse.ps1

Optional:
   powershell -ExecutionPolicy Bypass -File .\run_ml4t_ch7_per_ticker_rmse.ps1 -Ticker AAPL
   powershell -ExecutionPolicy Bypass -File .\run_ml4t_ch7_per_ticker_rmse.ps1 -BuildYahooAssets -RunNotebooks -Ticker AAPL
   powershell -ExecutionPolicy Bypass -File .\run_ml4t_ch7_per_ticker_rmse.ps1 -RunNotebooks
   powershell -ExecutionPolicy Bypass -File .\run_ml4t_ch7_per_ticker_rmse.ps1 -Python py -3

Outputs:
   ml4t_windows_one_click\out\per_ticker_rmse.csv
   ml4t_windows_one_click\out\per_date_rmse.csv
   ml4t_windows_one_click\out\model_summary.csv
   ml4t_windows_one_click\out\predictions.csv
   ml4t_windows_one_click\out\run_summary.txt

Chapter 17 reproduction:
   powershell -ExecutionPolicy Bypass -File .\run_ml4t_ch17_reproduce.ps1

If native Windows Zipline dependencies are unavailable, reproduce training and
NN predictions first:
   powershell -ExecutionPolicy Bypass -File .\run_ml4t_ch17_reproduce.ps1 -SkipBacktest

The Chapter 17 script reuses existing local files when present:
   machine-learning-for-trading\data\assets.h5
   machine-learning-for-trading\data\wiki_prices.csv
   machine-learning-for-trading\12_gradient_boosting_machines\data.h5
   machine-learning-for-trading\17_deep_learning\results\scores.h5
   machine-learning-for-trading\17_deep_learning\results\test_preds.h5

Notes:
- The script scans existing HDF/parquet datasets under machine-learning-for-trading.
- If no model dataset exists, run with -RunNotebooks or manually execute Chapter 7 notebooks/data setup.
- Core/data-reader dependencies are installed automatically if missing:
  numpy, pandas, scipy, scikit-learn, tables, pyarrow, fastparquet.
- -RunNotebooks additionally auto-installs:
  jupyter, nbconvert, nbformat, matplotlib, seaborn, pandas-datareader, yfinance, TA-Lib, statsmodels.
- -BuildYahooAssets creates machine-learning-for-trading\data\assets.h5 from Yahoo Finance.
  This is a compatibility dataset for running the notebooks when original Quandl WIKI
  prices are unavailable; it is not the official Quandl WIKI raw dataset.
- It does not fabricate results; if the book data are unavailable, it fails with a clear message.
