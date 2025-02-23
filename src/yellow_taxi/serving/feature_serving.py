from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


class FeatureServing:
    def __init__(self, feature_table_name: str, feature_spec_name: str, endpoint_name: str):
        """
        Initializes the Prediction Serving Manager.
        """
        self.feature_table_name = feature_table_name
        self.workspace = WorkspaceClient()
        self.feature_spec_name = feature_spec_name
        self.endpoint_name = endpoint_name
        self.online_table_name = f"{self.feature_table_name}_online"
        self.fe = feature_engineering.FeatureEngineeringClient()

    def create_online_table(self):
        """
        Creates an online table based on the feature table.
        """
        spec = OnlineTableSpec(
            primary_key_columns=["trip_id"],
            source_table_full_name=self.feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )
        self.workspace.online_tables.create(name=self.online_table_name, spec=spec)

    def create_feature_spec(self):
        """
        Creates a feature spec to enable feature serving.
        """
        features = [
            FeatureLookup(
                table_name=self.feature_table_name,
                lookup_key="trip_id",
                feature_names=["trip_distance", "passenger_count", "payment_type_discount", "Predicted_TaxiAmount"],
            )
        ]
        self.fe.create_feature_spec(name=self.feature_spec_name, features=features, exclude_columns=None)

    def deploy_or_update_serving_endpoint(self, workload_size: str = "Small", scale_to_zero: bool = True):
        """
        Deploys the feature serving endpoint in Databricks.
        :param workload_seze: str. Workload size (number of concurrent requests). Default is Small = 4 concurrent requests
        :param scale_to_zero: bool. If True, endpoint scales to 0 when unused.
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())

        served_entities = [
            ServedEntityInput(
                entity_name=self.feature_spec_name, scale_to_zero_enabled=scale_to_zero, workload_size=workload_size
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
