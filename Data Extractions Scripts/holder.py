import os
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_with_selenium(url):
    # Set up the WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run Chrome in headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    # Wait for the page to load
    driver.implicitly_wait(10)

    # Get the page source and parse it
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find the specific section using data-testid
    section = soup.find('section', {'data-testid': 'holders-top-institutional-holders'})
    if not section:
        print('Section not found')
        driver.quit()
        return None

    # Find the table within the section
    table_container = section.find('div', class_='tableContainer')
    if not table_container:
        print('Table container not found')
        driver.quit()
        return None

    table = table_container.find('table')
    if not table:
        print('Table not found')
        driver.quit()
        return None

    # Extract headers
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    print(f"Headers: {headers}")

    # Extract rows
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        print(f"Row data: {cells}")
        rows.append(cells)

    # Create a list of dictionaries
    data = [dict(zip(headers, row)) for row in rows]

    # Close the WebDriver
    driver.quit()

    return data

def main():
    # Define the tickers array
    tickers = ['SMCI']

    # Create the base directory for storing data
    base_dir = '/Users/abhishekjoshi/Documents/GitHub/stock_forecasting_CAI/Data'
    os.makedirs(base_dir, exist_ok=True)

    for ticker in tickers:
        # Construct the URL
        url = f'https://finance.yahoo.com/quote/{ticker}/holders/'

        # Scrape the data
        data = scrape_with_selenium(url)
        if data is not None:
            # Create a directory for the ticker
            ticker_dir = os.path.join(base_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)

            # Save the data to a JSON file
            json_filename = os.path.join(ticker_dir, f'{ticker}_holder.json')
            with open(json_filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            print(f"Data for {ticker} saved to {json_filename}")
        else:
            print(f"Failed to scrape data for {ticker}")

if __name__ == "__main__":
    main()
