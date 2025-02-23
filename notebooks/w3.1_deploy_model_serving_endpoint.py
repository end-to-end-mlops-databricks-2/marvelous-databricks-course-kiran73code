# Databricks notebook source
# MAGIC %pip install yellow_taxi-0.0.1-py3-none-any.whl
# MAGIC %restart_python
# COMMAND ----------
import os
import time
from typing import Dict, List

import mlflow
import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from yellow_taxi.config import ProjectConfig
from yellow_taxi.serving.model_serving import ModelServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)


# COMMAND ----------
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
print(os.environ["DBR_TOKEN"])
print(os.environ["DBR_HOST"])
# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "yellow-taxi-model-serving"


# COMMAND ----------
# set the tracking and registry uri to access the model metadata(for local run)
mlflow.set_tracking_uri("databricks://dbc-4894232b-9fc5")
mlflow.set_registry_uri("databricks-uc://dbc-4894232b-9fc5")


# COMMAND ----------
# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.yellow_taxi_model_basic", endpoint_name=endpoint_name
)


# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body
required_columns = [
    "fare_amount",
    "passenger_count",
    "trip_distance",
    "payment_type_discount",
    "PULocationID",
    "transaction_month",
    "transaction_day",
    "transaction_hour",
    "transaction_year",
    "payment_type",
]


# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample 100 records from the training set
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# Call the endpoint with one sample record

"""
Each dataframe record in the request body should be list of json with columns looking like:

[
    {
        "fare_amount": 10.0,
        "passenger_count": 2,
        "trip_distance": 2.0,
        "payment_type_discount": 1.5,
        "PULocationID": 100,
        "transaction_month": 6,
        "transaction_day": 10,
        "transaction_hour": 5,
        "transaction_year": 2020,
        "payment_type": 1,
    }
]
"""


def call_endpoint(record: List[Dict]):
    """
    Calls the model serving endpoint with a given input record.
    """

    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


# COMMAND ----------
# Call the endpoint with one sample record
status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# "load test"

for i in range(len(dataframe_records)):
    call_endpoint(dataframe_records[i])
    time.sleep(0.2)
