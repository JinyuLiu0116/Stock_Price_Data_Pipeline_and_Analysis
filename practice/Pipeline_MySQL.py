#use API for real time stock price
# fetch stock data from source and format them, and then send to MySQL database
# need to create a brench in MySQL to store these data
# the next step is add apache-airflow to schedule the time from Monday to Friday, between the time that stock markt open and close
# last step analysis and visualize the data by using pandas and matplotlib libraries
!pip install apache_beam
!pip install mysql_connector_python

#  We are doing ETL !! extract -> transform -> store
import requests #Used to make HTTP requests to get stock data from an API.
import apache_beam as beam #Apache Beam is a unified model for defining both batch and streaming data-parallel processing pipelines.
from apache_beam.options.pipeline_options import PipelineOptions #Used to configure Apache Beam pipelines.
#from apache_beam.io.jdbc import WriteToMySQL we do not need this one, we use beam.ParDo instead
import json #Used to parse JSON responses
import mysql.connector # A MySQL driver to connect and interact with MySQL databases.

class FetchStockPrice(beam.DoFn): # this class drived from beam.DoFn(Do function), which fetches stock prices from the Alpha Vantage API.
  def process(self, element): # The method that gets called for each element in the input PCollection (a collection of data elements).
    symbol = element  # Represents a stock symbol
    api_key = '2jAIXI9Kte2V8N3EhXT6LPOoUwe_5tfkKxgyeHxSw8rA7DNPg'  #t the key we got from API class
    #url = f'https://raw.githubusercontent.com/JinyuLiu0116/Stock_Price_Data_Pipeline_and_Analysis/main/daily_IBM.csv?token=GHSAT0AAAAAACTVWJPIUNVLSX6UIHC66GMCZUYAHBQ/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}' 
    response = requests.get(url)
    data = response.json()  # Makes a request to the API and returns the JSON response as a list containing the data.
    return [data]

class ParseAndFormatData(beam.DoFn): # this class is to parses the stock data and formats it.
  # The process method will be called for each element in the input PCollection.
  def process(self, element): # The process method takes self (the instance of the class) and element (an input element from the PCollection) as arguments
    # This line retrieves the value associated with the key 'Time Series (3min)' from the element dictionary
    time_series = element.get('Time Series (5min)',{}) #If the key is not found, it returns an empty dictionary {}.
    formatted_data = [] # !!important!! This initializes an empty list named formatted_data to store the formatted stock price data.
    # This for loop iterates over each item in the time_series dictionary.
    for timestamp, values in time_series.items(): # timestamp is the key, representing the time of the stock price data.
      formatted_data.append({   # This dictionary is then appended to the formatted_data list.
          'timestamp': timestamp,
          'open': values['1. open'],
          'high': values['2. high'],
          'low' : values['3. low'],
          'close': values['4. close'],
          'volume': values['5. volume'],
      })
      return formatted_data

class WriteToMySQL(beam.DoFn): 
  def __init__(self, host, database, user, password):
      self.host = host
      self.database = database
      self.user = user
      self.password = password

  def process(self, element):
    connection = mysql.connector.connect(
        host = self.host,
        database = self.database,
        user = self.user,
        password = self.password
    )
    cursor = connection.cursor()

    insert_statement = (
        "INSERT INTO stock_prices "
        "(timestamp, open, high, low, close, volume) "
        "VALUES (%(timestamp)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)"
    )

    for data in element:
      cursor.execute(insert_statement, data)
      connection.commit()

    cursor.close()
    connection.close()

def run():
  options = PipelineOptions()
  with beam.Pipeline(options = options) as p:
    (p
     | 'Read symbols' >> beam.Create([('APPL'), ('WFC'),('IBM')])
     | 'Fetch stock prices' >> beam.ParDo(FetchStockPrice())
     | 'Parse and format data' >> beam.ParDo(ParseAndFormatData())
     | 'Write to MySQL' >> beam.ParDo(WriteToMySQL(
             host='127.0.0.1',
             database='Stock_Price',
             user='root',
             password='root'
         )))

if __name__ == '__main__':
    run()
