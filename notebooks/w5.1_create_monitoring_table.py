# Databricks notebook source
# MAGIC %python
# pip install ../.


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

import datetime
import itertools
import time

import pandas as pd
import requests
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, to_utc_timestamp
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from yellow_taxi.config import ProjectConfig
from yellow_taxi.data_processor import generate_synthetic_data
from yellow_taxi.monitoring import create_or_refresh_monitoring

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# COMMAND ----------


# Encode categorical and datetime variables
def preprocess_data(df):
    label_encoders = {}
    for coll in df.select_dtypes(include=["object", "datetime"]).columns:
        le = LabelEncoder()
        df[coll] = le.fit_transform(df[coll].astype(str))
        label_encoders[coll] = le
    return df, label_encoders


train_set, label_encoders = preprocess_data(train_set)

# Define features and target (adjust columns accordingly)
features = train_set.drop(columns=["total_amount"])
target = train_set["total_amount"]


# COMMAND ----------

# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(features, target)

# Identify the most important features
feature_importances = pd.DataFrame({"Feature": features.columns, "Importance": model.feature_importances_}).sort_values(
    by="Importance", ascending=False
)

print("Top 5 important features:")
print(feature_importances.head(5))


# COMMAND ----------


inference_data_skewed = generate_synthetic_data(train_set, drift=True, num_rows=200)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Tables and Update house_features_online

# COMMAND ----------

inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)

# COMMAND ----------


workspace = WorkspaceClient()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Data to the Endpoint

# COMMAND ----------

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

test_set = (
    spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    .withColumn("trip_id", col("trip_id").cast("string"))
    .toPandas()
)


inference_data_skewed = (
    spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")
    .withColumn("trip_id", col("trip_id").cast("string"))
    .toPandas()
)


# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------


workspace = WorkspaceClient()

# Required columns for inference
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


# Sample records from inference datasets
sampled_skewed_records = inference_data_skewed[required_columns].to_dict(orient="records")
test_set_records = test_set[required_columns].to_dict(orient="records")

# COMMAND ----------


# Two different way to send request to the endpoint
# 1. Using https endpoint
def send_request_https(dataframe_record):
    model_serving_endpoint = f"https://{host}/serving-endpoints/yellow-taxi-model-serving-fe/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response


# 2. Using workspace client
def send_request_workspace(dataframe_record):
    response = workspace.serving_endpoints.query(
        name="yellow-taxi-model-serving-fe", dataframe_records=[dataframe_record]
    )
    return response


# COMMAND ----------

# Loop over test records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=20)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)


# COMMAND ----------

# Loop over skewed records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=30)
for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for skewed data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Refresh Monitoring

# COMMAND ----------


spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)
