# Databricks notebook source
# MAGIC %pip install --upgrade --quiet databricks-sdk mlflow
# MAGIC %restart_python

# COMMAND ----------

from typing import Any, Dict, Optional, List

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
model_config: ModelConfig = ModelConfig(development_config=model_config_file)

databricks_resources: Dict[str, Any] = model_config.get("databricks_resources")


scope_name: str = databricks_resources.get("scope_name")
secret_name: str = databricks_resources.get("secret_name")
client_id: str = databricks_resources.get("client_id")
client_secret: str = databricks_resources.get("client_secret")
users: List[str] = databricks_resources.get("users")

print(f"{scope_name=}")
print(f"{secret_name=}")
print(f"{client_id=}")
print(f"{client_secret=}")
print(f"{users=}")


# COMMAND ----------

from typing import List

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import SecretScope, AclPermission
from databricks.sdk.service.settings import CreateTokenResponse
from databricks.sdk.service.workspace import SecretMetadata


w: WorkspaceClient = WorkspaceClient(
    client_id=dbutils.secrets.get(scope_name, client_id), 
    client_secret=dbutils.secrets.get(scope_name, client_secret)
)

scopes: List[SecretScope] = w.secrets.list_scopes()
if not any([scope.name == scope_name for scope in scopes]):
    print(f"Creating scope {scope_name}")
    w.secrets.create_scope(scope_name)
else:
    print(f"Scope {scope_name} already exists")


secrets: List[SecretMetadata] = w.secrets.list_secrets(scope_name)
if not any([secret.key == secret_name for secret in secrets]):
    lifetime_seconds: int = None #60*60*24*180 # 180 days
    print(f"Creating secret {secret_name}")
    create_token_response: CreateTokenResponse = w.tokens.create(comment="Databricks Access Token for PAWS POC", lifetime_seconds=lifetime_seconds)
    w.secrets.put_secret(scope_name, secret_name, string_value=create_token_response.token_value)
    w.secrets.put_acl(scope_name, w.current_user.me().user_name, AclPermission.MANAGE)
    for user in users:
        w.secrets.put_acl(scope_name, user, AclPermission.MANAGE)
else:
    print(f"Secret {secret_name} already exists")

# COMMAND ----------

w.secrets.delete_secret(scope_name, secret_name)
