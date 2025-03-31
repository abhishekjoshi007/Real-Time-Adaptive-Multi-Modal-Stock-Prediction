import os
import requests
import json
import time
from datetime import datetime
import pandas as pd
#For Progress bar
from tqdm import tqdm 
# For cleaning HTML tags
import re  

api_url = "https://api-2-0.spot.im/v1.0.0/conversation/read"
csv_path = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/CSV/final_output_with_conversation_id_1.csv'
base_dir = 'filtered_comments'

def convert_timestamp(unix_timestamp):
    """Convert Unix timestamp to human-readable date."""
    try:
        return datetime.utcfromtimestamp(unix_timestamp).strftime('%Y-%m-%d')
    except Exception as e:
        print(f"DEBUG: Invalid timestamp {unix_timestamp} - {e}")
        return None

def clean_html_tags(text):
    """Remove HTML tags like <p>, <br>, etc., from a string."""
    clean_text = re.sub(r'<[^>]*>', '', text)  # Regex to remove HTML tags
    return clean_text.strip()

def clean_comment_data(comment):
    """Clean the comment data into the desired format."""
    cleaned_data = {
        "time": convert_timestamp(comment.get("time")),
        "replies_count": comment.get("replies_count", 0),
        "rank": {
            "ranks_up": comment.get("rank", {}).get("ranks_up", 0),
            "ranks_down": comment.get("rank", {}).get("ranks_down", 0),
            "ranked_by_current_user": comment.get("rank", {}).get("ranked_by_current_user", 0)
        },
        "replies": [],
        "content": [],
        "best_score": comment.get("best_score", 0),
        "total_replies_count": comment.get("total_replies_count", 0),
        "user_reputation": comment.get("user_reputation", 0),
        "additional_data": comment.get("additional_data", {})  # Add `additional_data` if available
    }

    # Process replies recursively
    if "replies" in comment and comment["replies"]:
        for reply in comment["replies"]:
            cleaned_data["replies"].append(clean_comment_data(reply))

    # Process main content
    if "content" in comment and comment["content"]:
        for content_item in comment["content"]:
            cleaned_data["content"].append({"text": clean_html_tags(content_item.get("text", ""))})

    return cleaned_data

def fetch_comments_within_date_range(payload, headers, start_date, end_date):
    """
    Fetch comments within the given date range and clean the data as we fetch.
    """
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    cleaned_comments = []

    try:
        while True:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            if response.status_code != 200:
                print(f"Failed to fetch data: Status code {response.status_code}")
                break

            data = response.json()
            batch_comments = data.get("conversation", {}).get("comments", [])

            # If no comments are returned, stop the process
            if not batch_comments:
                break

            for comment in batch_comments:
                if 'written_at' in comment:
                    comment_date = datetime.utcfromtimestamp(comment['written_at'])
                    if start_date_obj <= comment_date <= end_date_obj:
                        # Clean and append the comment
                        cleaned_comments.append(clean_comment_data(comment))
                    elif comment_date < start_date_obj:
                        # Stop fetching further if we encounter a comment older than `start_date`
                        print(f"DEBUG: Stopping fetch, comment date {comment_date} is before {start_date}.")
                        return cleaned_comments

            # If there are no more comments to fetch
            if not data.get("conversation", {}).get("has_next", False):
                break

            # Update the offset for the next batch
            payload["offset"] = data["conversation"]["offset"]
            print(f"Fetched {len(cleaned_comments)} cleaned comments so far within date range...")
            time.sleep(1)  # Pause to respect rate limits
    except Exception as e:
        print(f"An error occurred while fetching comments: {e}")
    return cleaned_comments

def main():
    data = pd.read_csv(csv_path)
    total_tickers = len(data)

    os.makedirs(base_dir, exist_ok=True)

    start_date = '2024-08-01'
    end_date = '2024-10-31'

    for idx, row in tqdm(data.iterrows(), total=total_tickers, desc="Processing Tickers"):
        payload = {
            "conversation_id": row['Conversation Id'],
            "count": 100,  # Increase batch size to 100
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
        cleaned_comments = fetch_comments_within_date_range(payload, api_headers, start_date, end_date)

        ticker_dir = os.path.join(base_dir, row['Ticker'])
        os.makedirs(ticker_dir, exist_ok=True)

        filename = os.path.join(ticker_dir, f"{row['Ticker']}_comments_filtered.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cleaned_comments, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(cleaned_comments)} cleaned comments for {row['Ticker']} to {filename}")

if __name__ == "__main__":
    main()
