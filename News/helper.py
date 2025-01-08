import os

def rename_files(base_dir, ticker_array, stock_name_array):
    """
    Renames files in the directory structure to ensure all files start with the ticker name.

    :param base_dir: Base directory containing ticker folders.
    :param ticker_array: List of ticker names.
    :param stock_name_array: List of stock names corresponding to the tickers.
    """
    # Ensure ticker_array and stock_name_array have the same length
    if len(ticker_array) != len(stock_name_array):
        raise ValueError("Ticker array and stock name array must have the same length.")

    for ticker, stock_name in zip(ticker_array, stock_name_array):
        ticker_dir = os.path.join(base_dir, ticker)

        # Check if the ticker folder exists
        if not os.path.exists(ticker_dir):
            print(f"Ticker folder '{ticker}' does not exist. Skipping.")
            continue

        # Iterate through files in the ticker directory
        for file_name in os.listdir(ticker_dir):
            file_path = os.path.join(ticker_dir, file_name)

            # Skip if not a file
            if not os.path.isfile(file_path):
                continue

            # Rename the file if it doesn't start with the ticker name
            if not file_name.startswith(ticker):
                new_file_name = file_name.replace(stock_name, ticker, 1)
                new_file_path = os.path.join(ticker_dir, new_file_name)

                # Rename the file
                os.rename(file_path, new_file_path)
                print(f"Renamed '{file_name}' to '{new_file_name}' in '{ticker_dir}'.")

# Example usage
if __name__ == "__main__":
    base_directory = "/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data 2"  # Replace with the path to your "Data 2" directory

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
    stock_names = [
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

    rename_files(base_directory, tickers, stock_names)
