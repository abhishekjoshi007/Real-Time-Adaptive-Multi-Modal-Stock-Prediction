#This code works for both stock and crypto

import yfinance as yf
import pandas as pd
import os

# 1) READ TICKERS FROM CSV
#    Replace "coin_tickers.csv" with your actual filename/path
tickers_df = pd.read_csv("/Users/abhishekjoshi/Documents/GitHub/Cross-Market-Deep-Learning-Multi-Modal-Stock-Crypto-Prediction/CSV/Crypto.csv")  # CSV has columns like "Coin Name,Ticker,..."

# 2) CONFIGURATION
start_date = "2024-08-01"
end_date   = "2024-11-01"
base_folder_name = "/Users/abhishekjoshi/Documents/GitHub/Cross-Market-Deep-Learning-Multi-Modal-Stock-Crypto-Prediction/Combined Data /Historic Data Cry"  # Main folder

# Create base folder if it doesn't exist
if not os.path.exists(base_folder_name):
    os.makedirs(base_folder_name)

# 3) LOOP OVER EACH ROW, DOWNLOAD & SAVE DATA
for idx, row in tickers_df.iterrows():
    # Grab the ticker symbol from the CSV row
    ticker = row["Ticker"]  # e.g. "BTC-USD"

    print(f"Processing {ticker}...")

    # 3a) Download Data
    raw_data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        group_by="ticker",   # multi-index columns under ticker name
        auto_adjust=False    # includes both "Close" (raw) and "Adj Close"
    )

    # If nothing returned (e.g. invalid ticker or no data), skip
    if raw_data.empty:
        print(f"No data returned for {ticker}. Skipping.")
        continue

    # 3b) Flatten multi-index columns for single ticker
    #     Typically columns are: "Open", "High", "Low", "Close", "Adj Close", "Volume"
    try:
        data = raw_data[ticker].copy()  # If multi-index
    except KeyError:
        # If yfinance returns single-level columns
        data = raw_data.copy()

    # 3c) Convert Date from index to a normal column
    data.reset_index(inplace=True)

    # 3d) Rename "Adj Close" → "Price"
    if "Adj Close" in data.columns:
        data.rename(columns={"Adj Close": "Adj Close"}, inplace=True)

    # 3e) Insert a "Ticker" column
    if "Ticker" not in data.columns:
        data.insert(1, "Ticker", ticker)

    # 3f) Reorder columns: Date, Ticker, Open, High, Low, Close, Adj Close, Volume
    desired_order = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_cols = [col for col in desired_order if col not in data.columns]
    if missing_cols:
        print(f"Missing columns {missing_cols} for {ticker}. Skipping.")
        continue

    data = data[desired_order]

    # 4) SAVE TO SUBFOLDER NAMED AFTER THE TICKER
    #    e.g. "Historic DATA/BTC-USD/BTC-USD.csv"
    ticker_folder = os.path.join(base_folder_name, ticker)  # e.g. "Historic DATA/BTC-USD"
    os.makedirs(ticker_folder, exist_ok=True)

    # Filename = TICKER.csv (e.g. "BTC-USD.csv")
    filename = f"{ticker}.csv"
    filepath = os.path.join(ticker_folder, filename)
    data.to_csv(filepath, index=False)

    print(f"Saved {ticker} data to {filepath}")

print("Done!")