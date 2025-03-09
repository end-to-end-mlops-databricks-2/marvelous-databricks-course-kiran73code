from datetime import datetime, timedelta

import mlflow
import pandas as pd
import pytz
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from yellow_taxi.config import ProjectConfig, Tags


class FeatureLookUpModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.yellow_taxi_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_is_weekend"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self):
        """
        Create or replace the yellow_taxi_features table and populate it.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (payment_type BIGINT NOT NULL, payment_type_discount DOUBLE NOT NULL);
        """)
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT payment_type_pk PRIMARY KEY(payment_type);"
        )
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        logger.info("train_set table contain all payment_discount_type.")
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT DISTINCT(CAST(payment_type AS BIGINT)), CAST(payment_type_discount AS DOUBLE) FROM {self.catalog_name}.{self.schema_name}.train_set"
        )

        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self):
        """
        Define a function to calculate the weekend or not.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(transaction_day INT, transaction_month INT, transaction_year INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        from datetime import datetime, timedelta
        import pytz
        # Start date always consider 1st of the month
        start_date = datetime(transaction_year, transaction_month, 1, tzinfo=pytz.timezone('America/New_York'))
        day_number = transaction_day
        target_date = start_date + timedelta(days=day_number - 1)
        return 1 if target_date.weekday() >= 5 else 0
        $$
        """)
        logger.info("✅ Feature function defined.")

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        """
        # we droping the payment_discount_type column as we are going to use the feature lookup table
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "payment_type_discount"
        )

        # we are not removing the payment_discount_type column from the test set as we are going to use it for prediction
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        logger.info("✅ Data successfully loaded.")

    def is_weekend(self, transaction_day: int, transaction_month: int, transaction_year: int) -> int:
        """
        Calculate if the day is a weekend or not.
        """
        # Start date consider always 1st of the month
        start_date = datetime(transaction_year, transaction_month, 1, tzinfo=pytz.timezone("America/New_York"))
        target_date = start_date + timedelta(days=transaction_day - 1)
        return 1 if target_date.weekday() >= 5 else 0

    def feature_engineering(self):
        """
        Perform feature engineering by linking data with feature tables.
        """
        # Using feature engineering client to create training set it will automatically link extra features from the feature lookup table and feature function return value
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["payment_type_discount"],
                    lookup_key="payment_type",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="is_weekend",
                    input_bindings={
                        "transaction_day": "transaction_day",
                        "transaction_month": "transaction_month",
                        "transaction_year": "transaction_year",
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc", "trip_id"],
        )

        self.training_df = self.training_set.load_df().toPandas()

        # Calculate is_weekend for the test set bcz we are not using feature engineering client for the test set
        self.test_set["is_weekend"] = self.test_set.apply(
            lambda row: self.is_weekend(row["transaction_day"], row["transaction_month"], row["transaction_year"]),
            axis=1,
        )
        # training_df is feature engineered training set and  test set is normal table data without feature engineering
        self.X_train = self.training_df[self.num_features + self.cat_features + ["is_weekend"]]
        self.y_train = self.training_df[self.target]

        self.X_test = self.test_set[self.num_features + self.cat_features + ["is_weekend"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self):
        """
        Train the model and log results to MLflow.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(**self.parameters))])

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")

            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self):
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.yellow_taxi_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.yellow_taxi_model_fe",
            alias="latest-model",
            version=latest_version,
        )

    def load_latest_model_and_predict(self, X):
        """
        Load the trained model from MLflow using Feature Engineering Client and make predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.yellow_taxi_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions

    def update_feature_table(self):
        """
        Updates the yellow_taxi_features table with the latest records from train and test sets.
        1. Feature table is updated with the latest records from the train and test data only if it's contain the new payment type.
        (In my case, I have only 5 payment types, so I am updating the feature table only if I get a new payment type in future data.)
        """
        queries = [
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.train_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT payment_type, payment_type_discount
            FROM {self.config.catalog_name}.{self.config.schema_name}.train_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp) AND payment_type > 5
            """,
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.test_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT payment_type, payment_type_discount
            FROM {self.config.catalog_name}.{self.config.schema_name}.test_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp) AND payment_type > 5
            """,
        ]

        for query in queries:
            logger.info("Executing SQL update query...")
            self.spark.sql(query)
        logger.info("Yellow taxi feature paymet type discount is updated to feature table successfully.")

    def model_improved(self, test_set: pd.DataFrame):
        """
        Evaluate the model performance on the test dataset.
        """
        X_test = test_set.drop(self.config.target)
        test_set = test_set.select("trip_id", "total_amount")

        predictions_latest = self.load_latest_model_and_predict(X_test).withColumnRenamed(
            "prediction", "prediction_latest"
        )

        current_model_uri = f"runs:/{self.run_id}/lightgbm-pipeline-model-fe"
        predictions_current = self.fe.score_batch(model_uri=current_model_uri, df=X_test).withColumnRenamed(
            "prediction", "prediction_current"
        )

        logger.info("Predictions are ready.")

        # Join the DataFrames on the 'trip_id' column
        df = test_set.join(predictions_current, on="trip_id").join(predictions_latest, on="trip_id")

        # Calculate the absolute error for each model
        df = df.withColumn("error_current", F.abs(df["total_amount"] - df["prediction_current"]))
        df = df.withColumn("error_latest", F.abs(df["total_amount"] - df["prediction_latest"]))

        # calculate the mean squared error(MSE) for each model
        df = df.withColumn("mse_current", F.pow(df["total_amount"] - df["prediction_current"], 2))
        df = df.withColumn("mse_latest", F.pow(df["total_amount"] - df["prediction_latest"], 2))

        mse_current = df.agg(F.mean("mse_current")).collect()[0][0]
        mse_latest = df.agg(F.mean("mse_latest")).collect()[0][0]

        # Calculate the Mean Absolute Error (MAE) for each model
        mae_current = df.agg(F.mean("error_current")).collect()[0][0]
        mae_latest = df.agg(F.mean("error_latest")).collect()[0][0]

        # Compare models based on MSE
        logger.info(f"MSE for Current Model: {mse_current}")
        logger.info(f"MSE for Latest Model: {mse_latest}")

        # Compare models based on MAE
        logger.info(f"MAE for Current Model: {mae_current}")
        logger.info(f"MAE for Latest Model: {mae_latest}")

        if mae_current < mae_latest:
            logger.info("Current Model performs better. Registering new model.")
            return True

        elif mse_current < mse_latest:
            logger.info("Current Model performs better. Registering new model.")
            return True

        else:
            logger.info("New Model performs worse. Keeping the old model.")
            return False
