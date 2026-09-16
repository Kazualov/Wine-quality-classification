from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
        dag_id="wine_data_engineering",
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
) as dag:

    process_data = BashOperator(
        task_id="process_wine_data",
        bash_command=(
            "cd /opt/project && "
            "python code/datasets/preprocess.py"
        ),
    )

    process_data