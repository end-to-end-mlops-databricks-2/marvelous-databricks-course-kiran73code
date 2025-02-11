import logging

import pandas as pd
import yaml
from pyspark.sql import SparkSession

from yellow_taxi.config import ProjectConfig
from yellow_taxi.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

# df = spark.read.csv(
#     f"/Volumes/{config.catalog_name}/{config.schema_name}/yello_taxi/yellow_tripdata_2020-06.csv", header=True, inferSchema=True
# ).toPandas()
df = pd.read_csv("../data/yellow_tripdata_2020-06.csv")
logging.info("Data loaded successfully")


# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()


# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)


# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)


# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
