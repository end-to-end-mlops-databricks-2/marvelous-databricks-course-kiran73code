import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from yellow_taxi.config import ProjectConfig, Tags
from yellow_taxi.models.feature_lookup_model import FeatureLookUpModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Create feature table
fe_model.create_feature_table()


# Define is_weekend feature function
fe_model.define_feature_function()


# Load data
fe_model.load_data()


# Perform feature engineering
fe_model.feature_engineering()


# Train the model
fe_model.train()


# Train the model
fe_model.register_model()


# Lets run prediction on the last production model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop feature lookup columns and target
X_test = test_set.drop("payment_type_discount", config.target)


fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)

# Display predictions
logger.info(predictions)
