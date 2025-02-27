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
        # trip_id column is created for the purpose of tracking the records
        self.df["trip_id"] = self.df.index + 1
        self.df["trip_id"] = self.df["trip_id"].astype(str)

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
