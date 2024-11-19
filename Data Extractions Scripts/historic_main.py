import os
import requests
import json
import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm
# run from csv file final output
api_url = "https://api-2-0.spot.im/v1.0.0/conversation/read"
csv_path = 'Complete-List-of-SP-500-Index-Constituents-Apr-3-2024_1.csv'
base_dir = 'historic_data'

def convert_timestamp(unix_timestamp):
    """Convert Unix timestamp to human-readable date."""
    try:
        return datetime.utcfromtimestamp(unix_timestamp).strftime('%Y-%m-%d')
    except Exception as e:
        print(f"DEBUG: Invalid timestamp {unix_timestamp} - {e}")
        return None

def fetch_and_filter_comments(payload, headers, start_date, end_date):
    """Fetch comments from the API and filter them by date range."""
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    comments = []
    try:
        while True:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                print(f"Failed to fetch data: Status code {response.status_code}")
                break
            
            data = response.json()
            batch_comments = data.get("conversation", {}).get("comments", [])
            filtered_batch = []

            for comment in batch_comments:
                if 'written_at' in comment:
                    comment_date = datetime.utcfromtimestamp(comment['written_at'])
                    if start_date_obj <= comment_date <= end_date_obj:
                        comment['written_at_readable'] = convert_timestamp(comment['written_at'])
                        filtered_batch.append(comment)
                    elif comment_date < start_date_obj:
                        # Stop fetching as all subsequent comments will be older
                        return comments
                else:
                    print(f"DEBUG: Skipping comment without `written_at` field: {comment}")
            
            comments.extend(filtered_batch)

            # Break if no more comments or filtered batch is empty
            if not data.get("conversation", {}).get("has_next", False) or not filtered_batch:
                break
            
            # Update offset for next batch and avoid rate limits
            payload["offset"] = data["conversation"]["offset"]
            time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")

    return comments

def main():
    data = pd.read_csv(csv_path)
    total_tickers = len(data)

    os.makedirs(base_dir, exist_ok=True)

    # Define the date range
    start_date = '2024-08-01'
    end_date = '2024-10-31'

    for idx, row in tqdm(data.iterrows(), total=total_tickers, desc="Processing Tickers"):
        payload = {
            "conversation_id": row['Conversation Id'],
            "count": 25,
            "offset": 0,
            "sort_by": "newest",
        }
        api_headers = {
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json",
            "x-spot-id": row['X-Spot-Id'],
            "x-post-id": row['X-Post-Id'],
        }

        print(f"\nProcessing {idx + 1}/{total_tickers}: {row['Ticker']} ({row['Company Name']})")
        filtered_comments = fetch_and_filter_comments(payload, api_headers, start_date, end_date)

        ticker_dir = os.path.join(base_dir, row['Ticker'])
        os.makedirs(ticker_dir, exist_ok=True)

        filename = os.path.join(ticker_dir, f"{row['Ticker']}_comments_filtered.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(filtered_comments, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(filtered_comments)} filtered comments for {row['Ticker']} to {filename}")

if __name__ == "__main__":
    main()
