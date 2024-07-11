import apache_beam as beam
from apache_beam.options.pipline_options import PiplelineOptions 
#also we need to install 'requests' library this is requisted library along with apache-beam library
#Using the requests library in Apache Beam allows you to integrate external web services and APIs
#seamlessly into your data processing workflows, enabling rich and dynamic data interactions.
import requests
import json

#define a function to fetch stock data
def fetch_stock_data(symbol, api_key):
  url=f'https://raw.githubusercontent.com/JinyuLiu0116/Stock_Price_Data_Pipeline_and_Analysis/main/daily_IBM.csv'
  respones = requests.get(url)
  if respones.status_code=200:
    data=respones.json()
    return json.dumps(data)
  else:
    return None
#The symbol and api_key parameters are passed to the function
#The url string is constructed using an f-string, which replaces {symbol} with the actual stock symbol and {api_key} with your actual API key
#The constructed URL is used to make the request to the Alpha Vantage API, and the response is processed accordingly.

