import requests
import apache_beam as beam
import csv
from io import StringIO
import mysql.connector
from apache_beam.options.pipeline_options import PipelineOptions

def fetch_stock_data(symbol):
    api_key = '2jAIZCO9TtAToc1G9qAbS3Sd08K_4ua142296hvViJoeQcjE'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}'
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.read().decode('utf-8')
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")


def parse_csv(line):
    reader=csv.DictReader(StringIO(line))
    next(reader)
    
    for row in reader:#separeted the row #each row goes to pipelin
       try:
        #given back  the dictionary to pipeline 
        yield{
        'Date' :row['Date']  
        'Open' :float(row['Open'])
        'High' = float(row['High'])
        'Low' = float(row['Low'])
        'Close' = float(row['Close'])
        'Volume' = int(row['Volume'])
        }
    except ValueError:
        return None  e



class WriteToMySQL(beam.DoFn):
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def start_bundle(self):
        self.connection = mysql.connector.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password
        )
        self.cursor = self.connection.cursor()

    def process(self, element): 
        self.insert_statement = (
        "INSERT INTO stock_prices "
        "(symbol, timestamp, open, high, low, close, volume) "
        "VALUES (%(symbol)s, %(timestamp)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)" 
        )
        
            self.cursor.execute(self.insert_statement, data)
            self.connection.commit()

    def finish_bundle(self):
        self.cursor.close()
        self.connection.close()
    
            

def pipeline_run(symbol):
    pipeline_options = PipelineOptions()


    with beam.Pipeline(options=pipeline_options) as pipeline:
        stock_data = (
            pipeline
            |'Create Input'>>beam.create([symbol])
            | 'ReadData' >> beam.Map(fetch_stock_data)
            | 'ParseCSV' >> beam.FlatMap(parse_csv)
            | 'WriteToMySQL' >> beam.ParDo(WriteToMySQL(
                host = "localhost",
                user = "username",
               password = "password",
                database = "stock_data"))
        )

if __name__ == "__main__":


    symbol="IBM"
    
    pipeline_run(symbol)
