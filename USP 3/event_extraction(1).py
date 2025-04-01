import os
import pandas as pd
import spacy
from tqdm import tqdm

# Load SpaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Keywords for high-impact events
# Extended list of high-impact keywords
HIGH_IMPACT_KEYWORDS = [
    # Financial Events
    "earnings report", "quarterly results", "annual report", "revenue growth",
    "profit warning", "stock split", "dividend announcement", "buyback announcement",
    "IPO", "secondary offering", "shareholder meeting",
    # Corporate Events
    "merger", "acquisition", "joint venture", "partnership", "spin-off", "restructuring",
    "bankruptcy", "layoffs", "executive change", "CEO resignation", "leadership appointment",
    "board reshuffle",
    # Regulatory and Legal
    "policy change", "regulatory action", "antitrust investigation", "lawsuit",
    "patent infringement", "compliance issues", "government sanctions", "tax ruling",
    # Market and Economic
    "market volatility", "sector performance", "interest rate changes", "inflation reports",
    "GDP growth", "employment statistics", "industrial production", "consumer confidence",
    "federal reserve decision",
    # Product and Innovation
    "product launch", "product recall", "technological breakthrough", "innovation announcement",
    "service expansion", "research collaboration", "new patent",
    # Industry-Specific Events
    "supply chain issues", "trade tariffs", "energy prices", "natural disasters",
    "cybersecurity breach", "data leaks", "environmental concerns",
    # Geopolitical and Macro Events
    "political unrest", "geopolitical conflict", "trade agreement", "pandemic outbreak",
    "economic stimulus package", "natural disasters"
]


def extract_events_from_news(news_csv):
    """
    Extract events using keyword detection and Named Entity Recognition (NER) from news.csv.
    """
    # If the news file is empty or missing
    if os.stat(news_csv).st_size == 0:
        print(f"No news data in {news_csv}. Assuming no events.")
        return []
    
    news_data = pd.read_csv(news_csv)
    
    # Ensure the required columns exist
    required_columns = ['Date', 'News Title', 'News Abstract', 'News Content']
    if not all(col in news_data.columns for col in required_columns):
        raise ValueError(f"Missing required columns in news data. Expected columns: {required_columns}")
    
    events = []
    
    for date, group in tqdm(news_data.groupby("Date"), desc="Processing news by date"):
        cumulative_text = " ".join(
            group['News Title'].fillna("") + " " +
            group['News Abstract'].fillna("") + " " +
            group['News Content'].fillna("")
        )
        
        # Perform NLP processing
        doc = nlp(cumulative_text.lower())
        
        # Check for keyword matches
        keyword_detected = any(keyword in cumulative_text for keyword in HIGH_IMPACT_KEYWORDS)
        
        # Check for named entities related to stocks
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT']]
        
        # Add to events if any criteria is met
        if keyword_detected or entities:
            events.append({
                "Date": date,
                "Keywords": keyword_detected,
                "Entities": entities,
                "Event_Flag": 1,  # Mark as an event
            })
    
    return events

def merge_event_flags_with_historic_data(historic_data_csv, events):
    """
    Merge event flags with historical data based on the Date field.
    """
    # Load historical data
    df = pd.read_csv(historic_data_csv)
    if 'Date' not in df.columns:
        raise ValueError(f"'Date' column is missing in {historic_data_csv}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add the Event_Flag column with default value 0
    if 'Event_Flag' not in df.columns:
        df['Event_Flag'] = 0

    # Convert events to a DataFrame
    events_df = pd.DataFrame(events)
    
    if not events_df.empty:
        events_df['Date'] = pd.to_datetime(events_df['Date'])
        # Merge event flags with historical data
        df = df.merge(events_df[['Date', 'Event_Flag']], on='Date', how='left')
        # Fill missing Event_Flag values with 0
        df['Event_Flag'] = df['Event_Flag'].fillna(0).astype(int)
    
    return df


def process_stock_data(data_folder):
    """
    Process all stock folders to add event flags and integrate them with historical data.
    """
    for ticker in os.listdir(data_folder):
        ticker_folder = os.path.join(data_folder, ticker)
        
        # Check if required files exist
        news_csv = os.path.join(ticker_folder, f"{ticker}_news_url.csv")  # Updated to CSV
        historic_csv = os.path.join(ticker_folder, f"{ticker}_merged_with_vix.csv")
        
        if os.path.exists(news_csv) and os.path.exists(historic_csv):
            print(f"Processing {ticker}...")
            
            # Extract events
            events = extract_events_from_news(news_csv)
            
            # Merge event flags with historical data
            updated_df = merge_event_flags_with_historic_data(historic_csv, events)
            
            # Save updated data back to CSV
            updated_df.to_csv(historic_csv, index=False)
            print(f"Updated data saved for {ticker}.")
        else:
            print(f"Missing files for {ticker}, skipping...")

# Specify the root data folder
data_folder = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"

# Run the process
process_stock_data(data_folder)
