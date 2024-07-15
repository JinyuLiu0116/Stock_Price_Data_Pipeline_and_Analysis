import apache-airflow

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.empty import EmptyOperator

#Creating a DAG (Directed Acyclic Graph)
with DAG(
    dag_id="dag_name",
    start_date=datetime.datetime(2024, 1, 1),
    schedule="@daily",
);
#The schedule we can change it to daily, weekly, etc.

EmptyOperator(task_id="task", dag=my_dag)



#Default Constructors
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'stock_price': 132;
}

#Create DAG Object
dag = DAG(
    default_args = default_args,
    dag_id="dag_name",
    schedule_interval='@once', 
    catchup=False,
    description='First test code',
)

#Lists of tasks (work in progress) [task_id we put in the correct etl file]

first_task = PythonOperator(
    task_id='unknown_etl',
    python_callable=run_unknown_etl,
    dag=dag,
)


#Task Sequence (Below)

first_task >> [second_task, third_task]
[second_task, third_task] >> fourth_task



