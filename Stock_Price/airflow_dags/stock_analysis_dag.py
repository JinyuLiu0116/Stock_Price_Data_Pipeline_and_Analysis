import apache-airflow

from airflow import DAG
from airflow.operators.empty import EmptyOperator

with DAG(
    dag_id="dag_name",
    start_date=datetime.datetime(2024, 1, 1),
    schedule="@daily",
);
#The schedule we can change it to daily, weekly, etc.

EmptyOperator(task_id="task", dag=my_dag)


#Task Sequence (Below)

first_task >> [second_task, third_task]
[second_task, third_task] >> fourth_task

