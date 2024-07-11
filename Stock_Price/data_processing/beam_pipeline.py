import apache_beam as beam
import requests
import csv

#function to fetch CSV data from URL
def fetch_csv_data(url):
  response = resquests.get(url)
  response.raise_for_status() # ensure got a successful response
  return response.text
  
#Define a function to parse CSV rows
def parse_csv(line):
  return dict(zip(row.keys(), row))

#URL of the CSV file
url='https://raw.githubusercontent.com/JinyuLiu0116/Stock_Price_Data_Pipeline_and_Analysis/main/daily_IBM.csv'

# Fetch the CSV data from the URL
csv_data = fetch_csv_data(url)

#Define the pipeline
with beam.Pipeline() as pipeline:
  #read data from the CSV file
  lines = (
    pipeline
    |'Creat CSV data' >> beam.Create(csv_data.splitlines()[1:]) # to skip the header
    |'Parse CSV' >> beam.map(lambda line: next(csv.DictReader([line])))
    |'Formart as dictionary' >> beam.Map(parse_csv_line)
    | beam.io.WriteToText('output.txt')
  )
  # run the pipeline
pipeline.run().wait_until_finish()
