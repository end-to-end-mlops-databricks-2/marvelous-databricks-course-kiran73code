from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from yellow_taxi.config import ProjectConfig
from yellow_taxi.data_processor import DataProcessor


@pytest.fixture
def sample_df():
    data = {
        "tpep_pickup_datetime": ["2020-06-01 00:00:00", "2020-06-01 01:00:00"],
        "tpep_dropoff_datetime": ["2020-06-01 00:10:00", "2020-06-01 01:20:00"],
        "passenger_count": [1, 2],
        "trip_distance": [1.0, 2.5],
        "fare_amount": [10.0, 20.0],
        "PULocationID": [1, 2],
        "total_amount": [12.0, 22.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_config():
    config = Mock(spec=ProjectConfig)
    config.catalog_name = "test_catalog"
    config.schema_name = "test_schema"
    config.target = "total_amount"
    return config


@pytest.fixture
def mock_spark():
    spark = Mock()
    spark.createDataFrame = MagicMock()
    spark.sql = MagicMock()
    return spark


@pytest.fixture
def data_processor(sample_df, mock_config, mock_spark):
    return DataProcessor(sample_df, mock_config, mock_spark)


def test_preprocess(data_processor):
    data_processor.preprocess()
    df = data_processor.df

    assert "transaction_date" in df.columns
    assert "transaction_year" not in df.columns
    assert df["total_amount"].between(0, 200).all()
    assert df["PULocationID"].dtype == "object"


def test_split_data(data_processor):
    data_processor.preprocess()
    train_set, test_set = data_processor.split_data()

    assert len(train_set) > 0
    assert len(test_set) > 0
    assert len(train_set) + len(test_set) == len(data_processor.df)
