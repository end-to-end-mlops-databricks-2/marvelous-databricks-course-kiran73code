from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType


def create_or_refresh_monitoring(config, spark, workspace):
    inf_table = spark.sql(
        f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`yellow_taxi_fe_model_serving_payload`"
    )

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("fare_amount", DoubleType(), True),
                            StructField("passenger_count", DoubleType(), True),
                            StructField("trip_distance", DoubleType(), True),
                            StructField("PULocationID", StringType(), True),
                            StructField("transaction_month", IntegerType(), True),
                            StructField("transaction_day", IntegerType(), True),
                            StructField("transaction_hour", IntegerType(), True),
                            StructField("transaction_year", IntegerType(), True),
                            StructField("payment_type", IntegerType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    df_final = df_exploded.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
        "timestamp_ms",
        "databricks_request_id",
        "execution_time_ms",
        F.col("record.fare_amount").alias("fare_amount"),
        F.col("record.passenger_count").alias("passenger_count"),
        F.col("record.trip_distance").alias("trip_distance"),
        F.col("record.PULocationID").alias("PULocationID"),
        F.col("record.transaction_month").alias("transaction_month"),
        F.col("record.transaction_day").alias("transaction_day"),
        F.col("record.transaction_hour").alias("transaction_hour"),
        F.col("record.transaction_year").alias("transaction_year"),
        F.col("record.payment_type").alias("payment_type"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("yellow-taxi-model-fe").alias("model_name"),
    )

    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    inference_set_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")

    df_final_with_status = (
        df_final.join(
            test_set.select("transaction_month", "transaction_day", "transaction_hour", "total_amount"),
            on=["transaction_month", "transaction_day", "transaction_hour"],
            how="left",
        )
        .withColumnRenamed("total_amount", "total_amount_test")
        .join(
            inference_set_skewed.select("transaction_month", "transaction_day", "transaction_hour", "total_amount"),
            on=["transaction_month", "transaction_day", "transaction_hour"],
            how="left",
        )
        .withColumnRenamed("total_amount", "total_amount_inference")
        .select("*", F.coalesce(F.col("total_amount_test"), F.col("total_amount_inference")).alias("total_amount"))
        .drop("total_amount_test", "total_amount_inference")
        .withColumn("total_amount", F.col("total_amount").cast("double"))
        .withColumn("prediction", F.col("prediction").cast("double"))
        .dropna(subset=["total_amount", "prediction"])
    )

    yellow_taxi_features = spark.table(f"{config.catalog_name}.{config.schema_name}.yellow_taxi_features")

    df_final_with_features = df_final_with_status.join(yellow_taxi_features, on="payment_type", how="left")

    df_final_with_features = df_final_with_features.withColumn(
        "payment_type_discount", F.col("payment_type_discount").cast("double")
    )

    df_final_with_features.write.format("delta").mode("append").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")


def create_monitoring_table(config, spark, workspace):
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="total_amount",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
