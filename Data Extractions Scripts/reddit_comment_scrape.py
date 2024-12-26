#!pip install praw pandas datetime
#using colab enterprise
import praw


reddit = praw.Reddit(
    client_id="qMC5FCxaIkKR1of9AzkYgg",
    client_secret="2gVN2nq-gibv8cHAQghN1UXV4nbYGQ",
    user_agent="stockpred-v1"
)
import pandas as pd
from datetime import datetime


file_path = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/CSV/ticker_list copy.csv"  
df = pd.read_csv(file_path)


stock_names = df['Ticker'].tolist()


start_date = datetime(2024, 8, 1)
end_date = datetime(2024, 10, 31)

import random
import time
from praw.models import MoreComments


def to_unix_timestamp(date):
    return int(date.timestamp())


start_timestamp = to_unix_timestamp(start_date)
end_timestamp = to_unix_timestamp(end_date)


results = []

for stock in stock_names:
    subreddit = reddit.subreddit("all")
    query = f"{stock}" 
    print(f"Fetching comments for: {stock}")

    try:

        for submission in subreddit.search(query, sort="new", time_filter="all", limit=50):  
            if start_timestamp <= submission.created_utc <= end_timestamp:
                try:
                    submission.comments.replace_more(limit=5)  
                    for comment in submission.comments.list():
                        results.append({
                            "stock": stock,
                            "submission_title": submission.title,
                            "comment": comment.body,
                            "created_utc": comment.created_utc
                        })
                except Exception as e:
                    print(f"Error fetching comments for submission: {e}")

            time.sleep(random.uniform(2, 5))  

    except Exception as e:
        print(f"Error fetching submissions for stock {stock}: {e}")
        time.sleep(10) 

print("Scraping complete!")


comments_df = pd.DataFrame(results)

comments_df['created_date'] = pd.to_datetime(comments_df['created_utc'], unit='s')

output_path = "/content/reddit_stock_comments.csv"
comments_df.to_csv(output_path, index=False)
print(f"Comments saved to: {output_path}")
comments_df.head(10)