# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from yellow_taxi.config import ProjectConfig, Tags
from yellow_taxi.models.basic_model import BasicModel

# COMMAND ----------
# Default profile:
mlflow.set_tracking_uri("databricks://dbc-4894232b-9fc5")
mlflow.set_registry_uri("databricks-uc://dbc-4894232b-9fc5")

# Profile called "course"
# mlflow.set_tracking_uri("databricks://course")
# mlflow.set_registry_uri("databricks-uc://course")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------
# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
# Load data and prepare features
basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
basic_model.train()
basic_model.log_model()

# COMMAND ----------
# Retrieve the run_id for the current run and load the model
run_id = mlflow.search_runs(experiment_names=["/Shared/yellow-taxi-basic"], filter_string="tags.branch='week2'").run_id[
    0
]
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")


# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_list = basic_model.load_latest_model_and_predict(X_test)


# COMMAND ----------
# predicted values
predictions_list


# COMMAND ----------
