from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
import pandas as pd

# Configure WebDriver (Ensure you have the correct WebDriver for your browser)
driver = webdriver.Chrome()

# Open the webpage
#chamnge offset from 100- 400 for next page 
# also use filters for industries
driver.get("https://finance.yahoo.com/screener/predefined/sec-ind_ind-largest-equities_electronics-computer-distribution/?offset=0&count=100")

# Initialize an empty list to store all stock data
all_data = []

while True:
    # Allow the page to load
    time.sleep(3)

    # Parse the current page's table
    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("table", {"class": "W(100%)"})  # Adjust the class name as necessary
    rows = table.find("tbody").find_all("tr")
    
    for row in rows:
        cells = row.find_all("td")
        all_data.append({
            "Symbol": cells[0].text.strip(),
            "Name": cells[1].text.strip(),
            "Price": cells[2].text.strip(),
            "Change": cells[3].text.strip(),
            "% Change": cells[4].text.strip(),
            "Volume": cells[5].text.strip(),
            "Avg Vol (3M)": cells[6].text.strip(),
            "Market Cap": cells[7].text.strip(),
            "PE Ratio": cells[8].text.strip(),
            "52 Week Range": cells[9].text.strip()
        })

    # Try to find the "Next" button and click it
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, ".next-button-class")  # Replace with actual selector
        next_button.click()
    except:
        print("No more pages.")
        break

# Save the data
df = pd.DataFrame(all_data)
df.to_csv("Electronics & Computer Distribution", index=False)

# Quit the driver
driver.quit()

print("Dynamic scraping complete. Data saved to 'Electronics & Computer Distribution'")
