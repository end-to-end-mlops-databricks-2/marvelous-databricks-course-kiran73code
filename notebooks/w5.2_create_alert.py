# Databricks notebook source
# MAGIC %md
# MAGIC ### Create a query that checks the percentage of MAE being higher than 7000

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

srcs = w.data_sources.list()


alert_query = """
SELECT
  (SUM(CASE WHEN mean_absolute_error > 9.6 THEN 1 ELSE 0 END) * 100.0 / SUM(CASE WHEN mean_absolute_error IS NOT NULL AND NOT isnan(mean_absolute_error) THEN 1 ELSE 0 END)) AS percentage_higher_than_9_6
FROM mlops_dev.kirancr2.model_monitoring_profile_metrics"""


query = w.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=f"yellow-taxi-alert-query-{time.time_ns()}",
        warehouse_id=srcs[0].warehouse_id,
        description="Alert on yellow taxi price prediction model MAE Greater than 9.6",
        query_text=alert_query,
    )
)

alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(column=sql.AlertOperandColumn(name="MAE_higher_than_9.6")),
            op=sql.AlertOperator.GREATER_THAN,
            threshold=sql.AlertConditionThreshold(value=sql.AlertOperandValue(double_value=45)),
        ),
        display_name=f"yellow-taxi-mae-alert-{time.time_ns()}",
        query_id=query.id,
    )
)


# COMMAND ----------

# cleanup
w.queries.delete(id=query.id)
w.alerts.delete(id=alert.id)

# COMMAND ----------
