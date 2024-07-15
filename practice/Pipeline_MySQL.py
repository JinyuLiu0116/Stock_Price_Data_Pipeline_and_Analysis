#use API for real time stock price
# fetch stock data from source and format them, and then send to MySQL database
# need to create a brench in MySQL to store these data
# the next step is add apache-airflow to schedule the time from Monday to Friday, between the time that stock markt open and close
# last step analysis and visualize the data by using pandas and matplotlib libraries
!pip install apache_beam
!pip install mysql_connector_python

#  We are doing ETL !! extract -> transform -> store
import requests
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
#from apache_beam.io.jdbc import WriteToMySQL we do not need this one, we use beam.ParDo instead
import json
import mysql.connector

class FetchStockPrice(beam.DoFn):
  def process(self, element):
    symbol = element
    api_key = '2jAIXI9Kte2V8N3EhXT6LPOoUwe_5tfkKxgyeHxSw8rA7DNPg'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return [data]

class ParseAndFormatData(beam.DoFn):
  def process(self, element):
    time_series = element.get('Time Series (3min)',{})
    formatted_data = []
    for timestamp, values in time_series.items():
      formatted_data.append({
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
     | 'Read symbols' >> beam.Create(['AAPL','GOOGL','WFC','AMZ','META'])
     | 'Fetch stock prices' >> beam.ParDo(FetchStockPrice())
     | 'Parse and format data' >> beam.ParDo(ParseAndFormatData())
     | 'Write to MySQL' >> beam.ParDo(WriteToMySQL( #if this line code do not work, we can ask professor
             host='our_mysql_host',
             database='the_database',
             user='the_username',
             password='the_password'
         )))
    
if __name__ == '__main__':
    run()
