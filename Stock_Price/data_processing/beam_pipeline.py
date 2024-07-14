#this code has error, but it is the form to create a pipeline
import apache_beam as beam
import requests
import csv

#URL of the CSV file
url='https://raw.githubusercontent.com/JinyuLiu0116/Stock_Price_Data_Pipeline_and_Analysis/main/daily_IBM.csv'

#function to fetch CSV data from URL
def fetch_csv_data(url):
  response = requests.get(url)

  print('status code =', response.status_code)
  #check if the request was successful, if yes valuse ==200, if not, value > 200
if response.status_code == 200: #Parse the Json response
  data = response.json()
  print(data.keys())  
  #response.raise_for_status() # ensure got a successful response
  #return response.text
#Define a function to parse CSV rows
def parse_csv(line):
  return dict(zip(line.keys(), line))
# Fetch the CSV data from the URL
csv_data = fetch_csv_data(url)

#Define the pipeline
with beam.Pipeline() as pipeline: #read data from the CSV file
  lines = (
    pipeline
    |'Creat CSV data' >> beam.Create(csv_data.splitlines()[1:]) # to skip the header
    |'Parse CSV' >> beam.Map(lambda line: next(csv.DictReader([line])))
    |'Formart as dictionary' >> beam.Map(parse_csv)
    | beam.io.WriteToText('output.txt')
  )
  # run the pipeline
pipeline.run().wait_until_finish()
