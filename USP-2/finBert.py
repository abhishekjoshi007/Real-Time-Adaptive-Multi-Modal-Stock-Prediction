import json
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text



# Sentiment analysis function using FinBERT with segmentation
# Refined sentiment analysis function with proper truncation and padding
def analyze_sentiment_finbert(text, tokenizer, model, nlp, chunk_size=512, overlap=50):
    # Clean the text first
    text = clean_text(text)

    # Tokenize and ensure text is within the model's max length
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=chunk_size)["input_ids"][0]

    # Ensure no input exceeds the model's limit (512 tokens)
    if len(tokens) > chunk_size:
        tokens = tokens[:chunk_size]

    sentiment_scores = {"Positive": 0, "Neutral": 0, "Negative": 0}
    confidence_scores = []

    # Process the tokenized input in chunks
    for start_idx in range(0, len(tokens), chunk_size - overlap):
        end_idx = start_idx + chunk_size
        
        # Slice the tokenized chunk
        chunk = tokens[start_idx:end_idx]

        # Convert the chunk back to text and process with FinBERT
        inputs = tokenizer.decode(chunk, skip_special_tokens=True)
        result = nlp(inputs)[0]

        # Update sentiment counts and confidence scores
        sentiment_scores[result["label"]] += 1
        confidence_scores.append(result["score"])

    # Aggregate final results
    sentiment = max(sentiment_scores, key=sentiment_scores.get)
    score = sentiment_scores["Positive"] - sentiment_scores["Negative"]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    return sentiment, score, avg_confidence



# Process comments and calculate daily sentiment scores
def process_data(data, source, tokenizer, model, nlp):
    rows = []
    for entry in data:
        date = entry["Date"]

        # Process main content
        for content in entry.get("content", []):
            text = content.get("text")
            if text:
                sentiment, score, confidence = analyze_sentiment_finbert(text, tokenizer, model, nlp)
                rows.append({
                    "Source": source,
                    "Date": date,
                    "Text": text,
                    "Sentiment": sentiment,
                    "Score": score,
                    "Confidence": confidence,
                    "Type": "Main"
                })

        # Process replies
        for reply in entry.get("replies", []):
            text = reply.get("text")
            if text:
                sentiment, score, confidence = analyze_sentiment_finbert(text, tokenizer, model, nlp)
                rows.append({
                    "Source": source,
                    "Date": date,
                    "Text": text,
                    "Sentiment": sentiment,
                    "Score": score,
                    "Confidence": confidence,
                    "Type": "Reply"
                })

    return rows

# Calculate cumulative daily sentiment scores
def calculate_daily_scores(df):
    df["Date"] = pd.to_datetime(df["Date"])
    daily_scores = df.groupby(["Source", "Date"]).agg({
        'Score': 'sum',
        'Confidence': 'mean',
        'Text': 'count'  # Count of text entries
    }).reset_index()
    daily_scores.rename(columns={"Score": "Cumulative Score", "Text": "Text Count"}, inplace=True)
    daily_scores['Normalized Score'] = daily_scores['Cumulative Score'] / daily_scores['Text Count']
    return daily_scores

# Main function
def main():
    # Ticker list
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
    'VSAT', 'BASE', 'MAXN', 'PGY', 'TDY', 'PAR', 'MRVL', 'PLUS', 'GCT', 'BKSY',
    'FOUR'
]

    # Load FinBERT model and tokenizer
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)  # Use CPU

    base_path = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/data"

    # Process each ticker
    for ticker in tickers:
        print(f"Processing data for {ticker}...")
        json_file = os.path.join(base_path, ticker, f"{ticker}_comments.json")
        if not os.path.exists(json_file):
            print(f"File not found: {json_file}")
            continue

        with open(json_file, 'r') as file:
            data = json.load(file)

        # Process Reddit and Yahoo data
        reddit_rows = process_data(data.get("REDDIT", []), "Reddit", tokenizer, model, nlp)
        yahoo_rows = process_data(data.get("YAHOO", []), "Yahoo", tokenizer, model, nlp)

        # Combine results into a DataFrame
        df = pd.DataFrame(reddit_rows + yahoo_rows)

        # Calculate cumulative daily scores
        daily_scores = calculate_daily_scores(df)

        # Save results to CSV files
        output_dir = os.path.join(base_path, ticker)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f"{ticker}_detailed_scores.csv"), index=False)
        daily_scores.to_csv(os.path.join(output_dir, f"{ticker}_daily_scores.csv"), index=False)

        print(f"Results for {ticker} saved to {output_dir}")

if __name__ == "__main__":
    main()
