from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
import requests
import os

def testAPI():
    print('Testing API')

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

print_testAPI_task = PythonOperator(
    task_id='testAPI',
    python_callable=testAPI,
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
print_testAPI_task >> print_pipeline_task >> print_database_task


# Extra notes:
# 
# 
# We can use requests to collect data from links [via requests]
# response = requests.get('https://api.quotable.io/random')
# quote = response.json()['content']
# print('Quote of the day: "{}"'.format(quote))
#
#
#

