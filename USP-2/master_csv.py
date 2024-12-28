import pandas as pd
import os

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
    'FOUR'
]

base_path = '/Users/gurojaschadha/Downloads/data 2/'

master_df = pd.DataFrame()

for ticker in tickers:
    updated_file = os.path.join(base_path, f"{ticker}/{ticker}_historic_data_updated.csv")
    
    if os.path.exists(updated_file):
        ticker_df = pd.read_csv(updated_file)
        master_df = pd.concat([master_df, ticker_df], ignore_index=True)
        print(f"Merged data for {ticker}")
    else:
        print(f"Updated file for {ticker} not found, skipping...")

master_csv_path = os.path.join(base_path, "master_historic_data.csv")
master_df.to_csv(master_csv_path, index=False)
print(f"Master CSV file created: {master_csv_path}")

'''
#master file without null sentiment scores
import pandas as pd
import os

base_path = '/Users/gurojaschadha/Downloads/data 2/'
master_csv_path = os.path.join(base_path, "master_historic_data.csv")

master_df = pd.read_csv(master_csv_path)

cleaned_master_df = master_df[master_df['Sentiment Score'] != 0].dropna(subset=['Sentiment Score'])

final_master_csv_path = os.path.join(base_path, "master_historic_data_no_null.csv")
cleaned_master_df.to_csv(final_master_csv_path, index=False)

print(f"Final master CSV file created: {final_master_csv_path}")

'''
