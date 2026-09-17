from pathlib import Path

import pendulum

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator


#Project root directory
project_root = Path(__file__).resolve().parents[3]


with DAG(
        dag_id="wine_quality_pipeline",
        start_date=pendulum.datetime(
            2026,
            1,
            1,
            tz="UTC"
        ),

        #Run every 5 minutes
        schedule="*/5 * * * *",

        #Do not run old missed executions
        catchup=False,

        #Do not allow two pipeline runs at the same time
        max_active_runs=1,

        tags=["wine", "mlops"],
) as dag:

    #1. Data Engineering

    data_engineering = BashOperator(
        task_id="data_engineering",

        bash_command=f"""
            set -e

            cd "{project_root}"

            python code/datasets/preprocess.py
        """,
    )


    #2. Model Engineering

    model_engineering = BashOperator(
        task_id="model_engineering",

        bash_command=f"""
            set -e

            cd "{project_root}"

            python code/models/train.py
        """,
    )


    #3. Deployment

    deployment = BashOperator(
        task_id="deployment",

        bash_command=f"""
            set -e

            cd "{project_root}/code/deployment"

            docker compose up -d --build
        """,
    )


    #Pipeline order

    data_engineering >> model_engineering >> deployment