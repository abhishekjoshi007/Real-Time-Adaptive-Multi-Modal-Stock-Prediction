import csv
import json
import requests

ticker_name = "TSLA"  # Replace with the relevant ticker name

api_url = "https://api-2-0.spot.im/v1.0.0/conversation/read"
payload = {
    "conversation_id": "sp_Rba9aFpG_finmb$27444752",
    "count": 25,
    "offset": 0,
    "sort_by": "newest",
}

api_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0",
    "Content-Type": "application/json",
    "x-spot-id": "sp_Rba9aFpG",
    "x-post-id": "finmb$27444752",
}


response = requests.post(api_url, headers=api_headers, data=json.dumps(payload))
data = response.json()

comments = []
while len(comments) < 10 and data["conversation"]["has_next"]:
    comm = data["conversation"]["comments"]
    comments.extend(comm)
    
    if len(comments) >= 10:  
        break

    payload["offset"] = data["conversation"]["offset"]
    response = requests.post(api_url, headers=api_headers, data=json.dumps(payload))
    data = response.json()

comments = comments[:10]
comment_data = [{"ticker": ticker_name, "comment": c["content"]} for c in comments]

csv_file = "first_10_comments.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["ticker", "comment"])
    writer.writeheader()
    writer.writerows(comment_data)

print(f"First 10 comments saved to {csv_file}")
