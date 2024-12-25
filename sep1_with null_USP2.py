import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to preprocess and calculate Z-score and EMA
def preprocess_data(ticker):
    historical_data_path = f"/Users/gurojaschadha/Downloads/data2/{ticker}/{ticker}_historic_data.csv"
    stock_df = pd.read_csv(historical_data_path)

    stock_df['EMA_Volume'] = stock_df['Volume'].ewm(span=20, adjust=False).mean()
    stock_df['EMA_Volume_Std'] = stock_df['Volume'].ewm(span=20, adjust=False).std()
    stock_df['Z_Volume'] = (stock_df['Volume'] - stock_df['EMA_Volume']) / stock_df['EMA_Volume_Std']

    processed_data_path = f"/Users/gurojaschadha/Downloads/data2/{ticker}/{ticker}_processed_data.csv"
    stock_df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved for {ticker} at {processed_data_path}")

# Function to calculate sentiment scores
def calculate_sentiment_scores(ticker):
    sentiment_data_path = f"/Users/gurojaschadha/Downloads/data2/{ticker}/{ticker}_flattened_comments.csv"
    sentiment_df = pd.read_csv(sentiment_data_path)

    analyzer = SentimentIntensityAnalyzer()

    def calculate_sentiment_score(text):
        if pd.isnull(text):
            return 0  # No comments found
        sentiment = analyzer.polarity_scores(text)
        compound_score = sentiment['compound']
        if compound_score > 0:
            return 1  # Bullish
        elif compound_score < 0:
            return -1  # Bearish
        else:
            return 0  # Neutral

    sentiment_df['Sentiment_Score'] = sentiment_df['Text'].apply(calculate_sentiment_score)


    sentiment_scores_path = f"/Users/gurojaschadha/Downloads/data2/{ticker}/{ticker}_sentiment_scores_with_custom_scores.csv"
    sentiment_df.to_csv(sentiment_scores_path, index=False)
    print(f"Sentiment scores saved for {ticker} at {sentiment_scores_path}")

# Function to merge processed data and sentiment scores
def merge_sentiment_with_processed_data(ticker):
    processed_data_path = f"/Users/gurojaschadha/Downloads/data2/{ticker}/{ticker}_processed_data.csv"
    sentiment_scores_path = f"/Users/gurojaschadha/Downloads/data2/{ticker}/{ticker}_sentiment_scores_with_custom_scores.csv"

    processed_df = pd.read_csv(processed_data_path)
    sentiment_df = pd.read_csv(sentiment_scores_path)

    # Merge sentiment scores with processed stock data on Date
    merged_df = pd.merge(processed_df, sentiment_df[['Date', 'Sentiment_Score']], on='Date', how='left')

    # Drop the old 'Sentiment_Score_x' column if it exists
    if 'Sentiment_Score_x' in merged_df.columns:
        merged_df.drop(columns=['Sentiment_Score_x'], inplace=True)

    merged_data_path = f"/Users/gurojaschadha/Downloads/data 2/{ticker}/{ticker}_merged.csv"
    merged_df.to_csv(merged_data_path, index=False)

    print(f"Successfully merged sentiment scores for {ticker}. Merged file saved at {merged_data_path}")

def main():
    tickers = [
        'PD', 'HEAR', 'AAPL', 'PANW', 'ARRY', 'TEL', 'ARQQ', 'ANET', 'UI', 'ZM', 'AGYS', 'FSLR',
        'INOD', 'UBER', 'SNPS', 'ADI', 'FORM', 'PLTR', 'SQ', 'RELL', 'AMKR', 'CSIQ', 'KD', 'BL',
        'TXN', 'KLAC', 'INTC', 'APP', 'BILL', 'RELY', 'CDNS', 'MU', 'APLD', 'FIS', 'TOST', 'DDOG',
        'AMAT', 'PAYO', 'NTAP', 'FICO', 'TTD', 'ATOM', 'DMRC', 'ENPH', 'TWLO', 'BMI', 'BMBL',
        'MSTR', 'OLED', 'CRWD', 'SOUN', 'LYFT', 'PATH', 'QCOM', 'BAND', 'RUM', 'ESTC', 'DOCU',
        'U', 'SEDG', 'MSFT', 'HPE', 'COHR', 'MEI', 'ONTO', 'ACN', 'WK', 'HPQ', 'S', 'GEN', 'ORCL',
        'OUST', 'DELL', 'ALAB', 'ODD', 'IBM', 'ADP', 'AMD', 'WOLF', 'LRCX', 'PI', 'SMCI', 'PAGS',
        'STNE', 'NXT', 'ZS', 'WDAY', 'HUBS', 'BTDR', 'NOW', 'AI', 'AFRM', 'NTNX', 'CORZ', 'KN',
        'INTU', 'WDC', 'TASK', 'ACMR', 'APH', 'OKTA', 'NEON', 'DOX', 'AEHR', 'CSCO', 'NVMI',
        'CRSR', 'SMTC', 'KEYS', 'AVGO', 'OS', 'RAMP', 'NCNO', 'INSG', 'KOSS', 'AAOI', 'SNOW',
        'ADSK', 'PSFE', 'RUN', 'UIS', 'ASTS', 'ON', 'KPLT', 'ADBE', 'FLEX', 'CAMT', 'QXO', 'AIP',
        'VSAT', 'BASE', 'MAXN', 'NVDA', 'PGY', 'TDY', 'PAR', 'MRVL', 'PLUS', 'GCT', 'BKSY',
        'FOUR', 'NVDA'
    ]

    for ticker in tickers:
        print(f"Processing {ticker}...")
        preprocess_data(ticker)
        calculate_sentiment_scores(ticker)
        merge_sentiment_with_processed_data(ticker)
        print(f"Completed processing for {ticker}.\n")

if __name__ == "__main__":
    main()
