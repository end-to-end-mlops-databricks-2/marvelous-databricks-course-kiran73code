# Databricks notebook source
# MAGIC %pip install house_price-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import mlflow
import os
import time
from typing import Dict, List

import requests
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from yellow_taxi.config import ProjectConfig
from yellow_taxi.serving.fe_model_serving import FeatureLookupServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "yellow-taxi-model-serving-fe"



# COMMAND ----------
# set the tracking and registry uri to access the model metadata
mlflow.set_tracking_uri("databricks://dbc-4894232b-9fc5")
mlflow.set_registry_uri("databricks-uc://dbc-4894232b-9fc5")



# COMMAND ----------
# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{schema_name}.yellow_taxi_model_fe",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.yellow_taxi_features",
)

# Create the online table for house features
feature_model_server.create_online_table()

# COMMAND ----------
# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body
required_columns = [
    "fare_amount",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "transaction_month",
    "transaction_day",
    "transaction_hour",
    "transaction_year",
    "payment_type",
]


spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

logger.info(train_set.dtypes)
logger.info(dataframe_records[0])


# COMMAND ----------
# Call the endpoint with one sample record
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


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# "load test"

for i in range(len(dataframe_records)):
    call_endpoint(dataframe_records[i])
    time.sleep(0.2)