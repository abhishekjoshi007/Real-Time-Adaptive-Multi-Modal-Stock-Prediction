#sentimental analysis using finbert

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Preprocess text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Perform sentiment analysis with FinBERT, handling long text
def analyze_sentiment_finbert(text, chunk_size=512, stride=256):
    text = clean_text(text)
    tokens = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    input_ids = tokens["input_ids"][0]
    sentiment_scores = {"Positive": 0, "Neutral": 0, "Negative": 0}
    confidence_scores = []

    for i in range(0, len(input_ids), stride):
        chunk = input_ids[i:i + chunk_size]
        inputs = tokenizer.decode(chunk, skip_special_tokens=True)
        result = nlp(inputs)[0]
        sentiment_scores[result['label']] += 1
        confidence_scores.append(result['score'])

    sentiment = max(sentiment_scores, key=sentiment_scores.get)
    score = sentiment_scores["Positive"] - sentiment_scores["Negative"]
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    return sentiment, score, avg_confidence

# Process comments and calculate daily sentiment scores
def process_data(data, source):
    rows = []
    for entry in data:
        date = entry["Date"]

        # Process main content
        for content in entry.get("content", []):
            text = content.get("text")
            if text:
                sentiment, score, confidence = analyze_sentiment_finbert(text)
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
                sentiment, score, confidence = analyze_sentiment_finbert(text)
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

# Visualize daily sentiment scores
def visualize_daily_scores(daily_scores):
    plt.figure(figsize=(10, 6))
    for source in daily_scores['Source'].unique():
        subset = daily_scores[daily_scores['Source'] == source]
        plt.plot(subset['Date'], subset['Cumulative Score'], label=source)
    plt.title('Cumulative Daily Sentiment Scores')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Score')
    plt.legend()
    plt.show()

# Main function
def main():
    # Load JSON data
    json_file = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/data copy/PD/PD_comments.json"  # Replace with your JSON file path
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Process Reddit and Yahoo data
    reddit_rows = process_data(data.get("REDDIT", []), "Reddit")
    yahoo_rows = process_data(data.get("YAHOO", []), "Yahoo")

    # Combine results into a DataFrame
    df = pd.DataFrame(reddit_rows + yahoo_rows)

    # Calculate cumulative daily scores
    daily_scores = calculate_daily_scores(df)

    # Save results to CSV files
    df.to_csv("finbert_sentiment_analysis_detailed.csv", index=False)
    daily_scores.to_csv("finbert_daily_sentiment_scores.csv", index=False)

    # Visualize daily scores
    visualize_daily_scores(daily_scores)

    # Print summaries
    print("Detailed sentiment analysis saved to finbert_sentiment_analysis_detailed.csv")
    print("Daily cumulative sentiment scores saved to finbert_daily_sentiment_scores.csv")

if __name__ == "__main__":
    main()
