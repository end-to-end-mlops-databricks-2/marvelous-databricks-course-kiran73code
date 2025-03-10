# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from yellow_taxi.config import ProjectConfig, Tags
from yellow_taxi.models.custom_model import CustomModel

# COMMAND ----------
# Default profile:
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Profile called "course"
# mlflow.set_tracking_uri("databricks://course")
# mlflow.set_registry_uri("databricks-uc://course")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------
# Initialize model with the config path
custom_model = CustomModel(
    config=config, tags=tags, spark=spark, code_paths=["../dist/yellow_taxi-0.0.1-py3-none-any.whl"]
)

# COMMAND ----------
# Load data and prepare features
custom_model.load_data()
custom_model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
custom_model.train()
custom_model.log_model()

# COMMAND ----------
# Retrieve the run_id for the current run and load the model
run_id = mlflow.search_runs(experiment_names=["/Shared/yellow-taxi-custom"]).run_id[0]
print(run_id)
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-yellow-taxi-model")

# COMMAND ----------
# Retrieve dataset for the current run
custom_model.retrieve_current_run_dataset()


# COMMAND ----------
# Register model
custom_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_list = custom_model.load_latest_model_and_predict(X_test)
# COMMAND ----------
