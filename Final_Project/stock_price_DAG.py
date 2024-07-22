from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from airflow.utils.dates import days_ago
from datetime import datetime
import requests
import os

# Tasks + Functions currently only print Strings as I do not have the knowledge on how
# to create tasks that connect to other tools such as MySQL, etc. to control the schedule of tasks.

def stockAPI():
    print('Testing Stock API')

def extractAPI():
    print('Extracting API test')

def pipeline():
    print('Testing pipeline')

def database():
    print('Testing database')

dag = DAG(
    'Stock-Price',
    default_args={'start_date': days_ago(1)},
    schedule_interval='30 9 * * 1-5',
    catchup=False
)

print_stockAPI_task = PythonOperator(
    task_id='testStockAPI',
    python_callable=stockAPI,
    dag=dag
)

print_extractAPI_task = PythonOperator(
    task_id='extractAPI',
    python_callable=extractAPI,
    dag=dag
)

print_pipeline_task = PythonOperator(
    task_id='pipeline',
    python_callable=pipeline,
    dag=dag
)

print_database_task = PythonOperator(
    task_id='database',
    python_callable=database,
    dag=dag
)




# Set the dependencies between the tasks
print_stockAPI_task >> print_extractAPI_task >> print_pipeline_task >> print_database_task


# Extra Optional notes:

# We can use requests to collect data from links [via requests] if needed

# response = requests.get('https://api.quotable.io/random')
# quote = response.json()['content']
# print('Quote of the day: "{}"'.format(quote))


# I tried to use MySqlOperator as part of the tasks/functions but it does not work with my limited knowledge and time:
# It was specifically the mysql_conn_id and the sql lines that I do not have the knowledge to make it work.

# execute_query = MySqlOperator(
#    task_id='execute_sql',
#    sql='SELECT * FROM wellsfargo.yahoo_api LIMIT 10;',
#    mysql_conn_id='mysql_default', # The ID of the MySQL connection you configured
#    dag=dag,
# )
