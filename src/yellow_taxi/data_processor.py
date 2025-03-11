import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from yellow_taxi.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""
        # Select the required columns from the DataFrame
        self.df = self.df[
            [
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "passenger_count",
                "trip_distance",
                "fare_amount",
                "PULocationID",
                "total_amount",
                "payment_type",
            ]
        ]

        # Filter out rows with total_amount  between 0 and 200$
        self.df = self.df[(self.df["total_amount"] > 0) & (self.df["total_amount"] < 200)]

        # change the data type of the column
        self.df["PULocationID"] = self.df["PULocationID"].astype(str)

        # Convert the pickup and dropoff datetime columns to datetime type
        self.df["tpep_pickup_datetime"] = pd.to_datetime(self.df["tpep_pickup_datetime"])
        self.df["tpep_dropoff_datetime"] = pd.to_datetime(self.df["tpep_dropoff_datetime"])

        # Extract the year, month and day  from the pickup datetime
        self.df["transaction_date"] = pd.to_datetime(self.df["tpep_pickup_datetime"].dt.date)
        self.df["transaction_year"] = self.df["tpep_pickup_datetime"].dt.year
        self.df["transaction_month"] = self.df["tpep_pickup_datetime"].dt.month
        self.df["transaction_day"] = self.df["tpep_pickup_datetime"].dt.day
        self.df["transaction_hour"] = self.df["tpep_pickup_datetime"].dt.hour

        # Calculate the trip duration in minutes
        self.df["trip_duration"] = (
            self.df["tpep_dropoff_datetime"] - self.df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60

        # Filter the data for particuar year and month
        self.df = self.df[(self.df["transaction_year"] == 2020) & (self.df["transaction_month"] == 6)]

        # based on the payment type we are giving discount to the customers
        """
            1 = Credit card
            2 = Cash
            3 = No charge
            4 = Dispute
            5 = Unknown
            6 = Voided trip

            we are promoting the customers to use credit card so we are giving discount to the customers who are using credit card
            also we are giving discount to the customers who are paying with cash
            discount = 3$ / int(payment_type)
            based on the BASE DISCOUNT PRICE we are going to calculate the discount ,
            in future we can change the BASE DISCOUNT PRICE(use feature LOOKUP later)
            discount amount we are going to create a new column called 'payment_type_discount'
        """

        # convert the payment type to int and replace NaN with 7
        self.df["payment_type"] = self.df["payment_type"].fillna(7).astype(int)
        BASE_DISCOUNT_PRICE = 3

        # payment type  value 0 it's not valid payment so update to default payment type 7.
        self.df["payment_type"] = self.df["payment_type"].apply(lambda x: 7 if x == 0 else x)

        self.df["payment_type_discount"] = self.df["payment_type"].apply(
            lambda x: BASE_DISCOUNT_PRICE / int(x) if x < 7 else 0
        )

        # select the required columns for feature engineering
        categorical_columns = [
            "PULocationID",
            "transaction_month",
            "transaction_day",
            "transaction_hour",
            "transaction_year",
            "payment_type",
        ]
        numerical_columns = ["total_amount", "trip_distance", "fare_amount", "passenger_count", "payment_type_discount"]
        all_needed_columns = categorical_columns + numerical_columns
        self.df = self.df[all_needed_columns]

        # Handle missing values by removing rows with missing values
        self.df.dropna(inplace=True)

        # aggregate the data by grouping the data by 'PULocationID','transaction_date','transaction_month','transaction_day','transaction_hour'
        self.df["transaction_count"] = self.df.groupby(categorical_columns).count().reset_index()["total_amount"]

        # Extract the required columns from the DataFrame
        cat_features = [
            "PULocationID",
            "transaction_month",
            "transaction_day",
            "transaction_hour",
            "transaction_year",
            "payment_type",
        ]
        num_features = ["trip_distance", "fare_amount", "passenger_count", "payment_type_discount"]

        # Extract target and relevant features
        target = self.config.target
        relevant_columns = cat_features + num_features + [target]
        self.df = self.df[relevant_columns]
        self.df.reset_index(drop=True, inplace=True)

        # trip_id column is created for the purpose of tracking the records and feature lookup
        timestamp_base = int(time.time() * 1000)
        self.df["trip_id"] = [str(timestamp_base + i) for i in range(self.df.shape[0])]

    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


def generate_synthetic_data(df, drift=False, num_rows=10):
    """Generates synthetic data based on the distribution of the input DataFrame."""
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        # Generate synthetic data based on the column type

        # Generate synthetic data based on the numeric column type
        if pd.api.types.is_numeric_dtype(df[column]):
            # if columns contain data like year, month etc we need to generate min max distribution values.
            # synthetic_data[column] = np.random.randint(df[column].min(), df[column].max() + 1, num_rows)
            synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)

        # Generate synthetic data based on the categorical column type
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            unique_values = df[column].dropna().unique()
            probabilities = df[column].value_counts(normalize=True).reindex(unique_values).fillna(0).values
            synthetic_data[column] = np.random.choice(unique_values, num_rows, p=probabilities)

        # Generate synthetic data based on the datetime column type
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Convert relevant numeric columns to integers
    int_columns = {
        "payment_type",
        "DOLocationID",
        "RatecodeID",
    }
    for col in int_columns.intersection(df.columns):
        synthetic_data[col] = synthetic_data[col].astype(np.int32)

    # Convert relevant numeric columns to floats
    float_columns = {
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
    }
    for col in float_columns.intersection(df.columns):
        synthetic_data[col] = synthetic_data[col].astype(np.float64)

    # create sample drifted data for monitoring testing
    if drift:
        # Skew the top features to introduce drift
        top_features = ["fare_amount", "trip_distance"]  # Select top 2 features
        for feature in top_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 2

    return synthetic_data
