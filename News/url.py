import json
import time
import requests
import csv
import os
from datetime import datetime

API_BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
tickers = [
    "PagerDuty", "Turtle Beach Corporation", "Apple", "Palo Alto Networks",
    "Array Technologies", "TE Connectivity Ltd.", "Arqit Quantum", "Arista Networks",
    "Ubiquiti", "Zoom Video Communications", "Agilysys", "First Solar",
    "Innodata", "Uber Technologies", "Synopsys", "Analog Devices",
    "FormFactor", "Palantir Technologies", "Block", "Richardson Electronics, Ltd.",
    "Amkor Technology", "Canadian Solar", "Kyndryl Holdings", "BlackLine",
    "Texas Instruments Incorporated", "KLA Corporation", "Intel Corporation", "AppLovin Corporation",
    "Bill.com Holdings", "Remitly Global", "Cadence Design Systems",
    "Micron Technology", "Applied Digital Corporation", "Fidelity National Information Services",
    "Toast", "Datadog", "Applied Materials", "Payoneer Global",
    "NetApp", "Fair Isaac Corporation", "The Trade Desk", "Atomera Incorporated",
    "Digimarc Corporation", "Enphase Energy", "Twilio", "Badger Meter",
    "Bumble", "MicroStrategy Incorporated", "Universal Display Corporation",
    "CrowdStrike Holdings", "SoundHound AI", "Lyft", "UiPath",
    "QUALCOMM Incorporated", "Bandwidth", "Rumble", "Elastic N.V.", "DocuSign",
    "Unity Software", "SolarEdge Technologies", "Microsoft Corporation",
    "Hewlett Packard Enterprise Company", "Coherent Corp.", "Methode Electronics",
    "Onto Innovation", "Accenture plc", "Workiva", "HP", "SentinelOne",
    "Gen Digital", "Oracle Corporation", "Ouster", "Dell Technologies",
    "Alabama Aircraft Industries", "Oddity Tech Ltd.", "International Business Machines Corporation",
    "Automatic Data Processing", "Advanced Micro Devices", "Wolfspeed",
    "Lam Research Corporation", "Impinj", "Super Micro Computer", "PagSeguro Digital Ltd.",
    "StoneCo Ltd.", "Nextracker", "Zscaler", "Workday", "HubSpot",
    "Bitdeer Technologies Group", "ServiceNow", "C3.ai", "Affirm Holdings",
    "Nutanix", "Core Scientific", "Knowles Corporation", "Intuit",
    "Western Digital Corporation", "TaskUs", "ACM Research", "Amphenol Corporation",
    "Okta", "Neonode", "Amdocs Limited", "Aehr Test Systems", "Cisco Systems",
    "Nova Ltd.", "Corsair Gaming", "Semtech Corporation", "Keysight Technologies",
    "Broadcom", "OneSpan", "LiveRamp Holdings", "nCino", "Inseego Corp.",
    "Koss Corporation", "Applied Optoelectronics", "Snowflake", "Autodesk",
    "Paysafe Limited", "Sunrun", "Unisys Corporation", "AST SpaceMobile",
    "ON Semiconductor Corporation", "Katapult Holdings", "Adobe", "Flex Ltd.",
    "Camtek Ltd.", "Quixant plc", "Arteris", "Viasat", "Couchbase",
    "Maxeon Solar Technologies, Ltd.", "Pagaya Technologies Ltd.", "Teledyne Technologies Incorporated",
    "PAR Technology Corporation", "Marvell Technology", "ePlus", "GigaCloud Technology",
    "BlackSky Technology", "Shift4 Payments"
]

headers = {
    "Accept": "application/json"
}

print("Fetching data... Please be patient.")

# Base directory for storing data
BASE_DIR = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"


def fetch_data():
    for ticker in tickers:
        print(f"Fetching data for ticker: {ticker}")
        params = {
            'q': f"{ticker} Stock",
            'key': 'SGGWyPzdW8s0fI1NnGGQsWgPIgrwYw2j',
            'begin_date': '20240801',
            'end_date': '20241031',
            'sort': 'oldest'
        }

        rows = []
        current_page = 0

        try:
            # Build the initial URL
            request_url = (
                f"{API_BASE_URL}?q={params['q']}&begin_date={params['begin_date']}&end_date={params['end_date']}"
                f"&sort={params['sort']}&api-key={params['key']}&page={current_page}"
            )
            # Initial request to get metadata
            response = requests.get(request_url, headers=headers)
            initial_data = response.json()

            total_hits = initial_data['response']['meta']['hits']
            results_per_page = len(initial_data['response']['docs'])
            total_pages = -(-total_hits // results_per_page)  # Ceiling division

            print(f"Total hits to fetch for {ticker}: {total_hits}")
            print(f"Total pages to fetch for {ticker}: {total_pages}")

            # Fetch data for each page
            for current_page in range(total_pages):
                request_url = (
                    f"{API_BASE_URL}?q={params['q']}&begin_date={params['begin_date']}&end_date={params['end_date']}"
                    f"&sort={params['sort']}&api-key={params['key']}&page={current_page}"
                )
                response = requests.get(request_url, headers=headers)
                data = response.json()

                # Extract and store article data
                for row in data['response']['docs']:
                    publication_date = row.get('pub_date', '')
                    formatted_date = ''
                    if publication_date:
                        formatted_date = datetime.strptime(publication_date, "%Y-%m-%dT%H:%M:%S%z").strftime("%Y-%m-%d")

                    rows.append({
                        'Date': formatted_date,
                        'URL': row.get('web_url'),
                        'News Title': row.get('headline', {}).get('main'),
                        'News Abstract': row.get('abstract'),
                        'News Content': row.get('lead_paragraph')
                    })

                print(f"Page #{current_page + 1} for {ticker} fetched and processed.")

                # Add an increasing delay between requests (exponential backoff)
                delay = 2**3 * 2000
                time.sleep(delay / 1000)

        except Exception as error:
            print(f"Error retrieving data for ticker {ticker}: {str(error)}")

        # Create folder structure
        ticker_dir = os.path.join(BASE_DIR, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        # Save data to a CSV file
        csv_file_name = os.path.join(ticker_dir, f"{ticker}_news_url.csv")
        try:
            with open(csv_file_name, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Date', 'URL', 'News Title', 'News Abstract', 'News Content']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"[X] Results ({len(rows)} rows) for {ticker} have been saved to '{csv_file_name}'.")
        except Exception as error:
            print(f"Error saving data for ticker {ticker}: {str(error)}")


# Run the fetch_data function
if __name__ == "__main__":
    fetch_data()
