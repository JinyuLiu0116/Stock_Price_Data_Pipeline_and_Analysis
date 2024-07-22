import requests
import csv
from datetime import datetime

def apidata_into_csv(api_url, api_key, stock_symbol, csv_filename, interval='15min'):
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'apikey': '2jAIZCO9TtAToc1G9qAbS3Sd08K_4ua142296hvViJoeQcjE',
        'symbol': 'CRWD', #WFC, MSFT
        'interval': '15min',
        'outputsize': 'full'
    }

    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        data = response.json()
        time_series_key = f'Time Series ({interval})'
        time_series_data = data.get(time_series_key, {})

        if not time_series_data:
            print("No data found.")
            return None

#filtering only for july data
        july = {date: metrics for date, metrics in time_series_data.items() if datetime.strptime(date, '%Y-%m-%d %H:%M:%S').month == 7}

# writting data to CSV
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            for date, metrics in july.items():
                csv_writer.writerow([
                    date,
                    metrics.get('1. open', ''),
                    metrics.get('2. high', ''),
                    metrics.get('3. low', ''),
                    metrics.get('4. close', ''),
                    metrics.get('5. volume', '')
                ])
        return csv_filename
    else:
        print(f"Failed to fetch data from API. Status code: {response.status_code}")
        return None

# saving CSV
if __name__ == "__main__":
    api_url = 'https://www.alphavantage.co/query'
    api_key = '2jAIZCO9TtAToc1G9qAbS3Sd08K_4ua142296hvViJoeQcjE'
    stock_symbol = 'CRWD'
    csv_filename = 'crdwstock_data.csv'
    interval = '15min'

    csv_dir = apidata_into_csv(api_url, api_key, stock_symbol, csv_filename, interval)
    if csv_dir:
        print(f"CSV file saved as: {csv_dir}")
    else:
        print("Failed to fetch and save data as CSV.")


        def split_row(row):
    return row.split(',')

def extract_time(row):
    date, open_price, high, low, close, volume = row
    time = date.split(' ')[1] if ' ' in date else ''
    return [date, time, open_price, high, low, close, volume]

def format_as_csv(row):
    return ','.join(row)

def run(input_csv, output_csv):
    pipeline_options = PipelineOptions()
    with beam.Pipeline(options=pipeline_options) as p:
        lines = (
            p
            | 'Read CSV' >> beam.io.ReadFromText(input_csv, skip_header_lines=1)
            | 'Split Rows' >> beam.Map(split_row)
            | 'Extract Time' >> beam.Map(extract_time)
            | 'Format as CSV' >> beam.Map(format_as_csv)
        )

        header = 'Date,Time,Open,High,Low,Close,Volume'

        # Create a PCollection containing the header
        header_pcollection = p | 'Create Header' >> beam.Create([header])

        # Write header and data to the output CSV
        (
            (header_pcollection, lines)
            | 'Combine Header and Data' >> beam.Flatten()
            | 'Write CSV' >> beam.io.WriteToText(output_csv, file_name_suffix='.csv', shard_name_template='')
        )

if __name__ == '__main__':
    input_csv = '/content/crdwstock_data.csv'
    output_csv = '/content/trans_crdwstock_data'
    run(input_csv, output_csv)


