# Databricks notebook source
# MAGIC %pip install --upgrade databricks-sdk databricks-vectorsearch mlflow databricks-langchain
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from typing import Any, Dict, Optional

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
model_config: ModelConfig = ModelConfig(development_config=model_config_file)

databricks_resources: Dict[str, Any] = model_config.get("databricks_resources")
retriever_config: Dict[str, Any] = model_config.get("retriever_config")
schema: Dict[str, Any] = retriever_config.get("schema")


catalog_name: str = databricks_resources.get("catalog_name")
database_name: str = databricks_resources.get("database_name")
source_table_name: str = retriever_config.get("source_table_name")
vector_search_index: str = retriever_config.get("vector_search_index")
vector_search_endpoint_name: str = databricks_resources.get("vector_search_endpoint_name")
primary_key: str = schema.get("primary_key")
embedding_source_column: str = schema.get("embedding_source_column")
embedding_model_endpoint_name: str = databricks_resources.get("embedding_model_endpoint_name")

secret_scope: str = databricks_resources.get("scope_name")
secret_name: str = databricks_resources.get("secret_name")
client_id: str = databricks_resources.get("client_id")
client_secret: str = databricks_resources.get("client_secret")

print(f"{secret_scope=}")
print(f"{secret_name=}")
print(f"{client_id=}")
print(f"{client_secret=}")

print(f"{catalog_name=}")
print(f"{database_name=}")
print(f"{source_table_name=}")
print(f"{vector_search_endpoint_name=}")
print(f"{vector_search_index=}")
print(f"{primary_key=}")
print(f"{embedding_source_column=}")
print(f"{embedding_model_endpoint_name=}")


# COMMAND ----------

import os

context = dbutils.entry_point.getDbutils().notebook().getContext()

workspace_host: str = spark.conf.get("spark.databricks.workspaceUrl")

os.environ["DATABRICKS_HOST"] = f"https://{workspace_host}"
#os.environ["DATABRICKS_TOKEN"] =  dbutils.secrets.get(secret_scope, secret_name) #or token
os.environ["DATABRICKS_CLIENT_ID"] = dbutils.secrets.get(secret_scope, client_id)
os.environ["DATABRICKS_CLIENT_SECRET"] = dbutils.secrets.get(secret_scope, client_secret)

print(f"workspace_host: {workspace_host}")


# COMMAND ----------

workspace_url: str = os.environ.get("DATABRICKS_HOST")
client_id: str = os.environ.get("DATABRICKS_CLIENT_ID")
client_secret: str = os.environ.get("DATABRICKS_CLIENT_SECRET")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import CatalogInfo, SchemaInfo

import app.catalog 

w: WorkspaceClient = WorkspaceClient(client_id=client_id, client_secret=client_secret)

catalog: CatalogInfo 
try:
  catalog = w.catalogs.get(catalog_name)
except Exception as e: 
  catalog = w.catalogs.create(catalog_name)

schema: SchemaInfo
try:
  schema = w.schemas.get(f"{catalog.full_name}.{database_name}")
except Exception as e:
  schema = w.schemas.create(database_name, catalog.full_name)
  
print(schema.full_name)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

from app.utils import endpoint_exists, wait_for_vs_endpoint_to_be_ready

                                    
vsc: VectorSearchClient = (
    VectorSearchClient(
        workspace_url=workspace_url, 
        service_principal_client_id=client_id,
        service_principal_client_secret=client_secret
    )
)

if not endpoint_exists(vsc, vector_search_endpoint_name):
    vsc.create_endpoint(name=vector_search_endpoint_name, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, vector_search_endpoint_name)

print(f"Endpoint named {vector_search_endpoint_name} is ready.")


# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.vector_search.index import VectorSearchIndex


from app.utils import index_exists, wait_for_index_to_be_ready

if not index_exists(vsc, vector_search_endpoint_name, vector_search_index):
  print(f"Creating index {vector_search_index} on endpoint {vector_search_endpoint_name}...")
  vsc.create_delta_sync_index(
    endpoint_name=vector_search_endpoint_name,
    index_name=vector_search_index,
    source_table_name=source_table_name,
    pipeline_type="TRIGGERED",
    primary_key=primary_key,
    embedding_source_column=embedding_source_column, #The column containing our text
    embedding_model_endpoint_name=embedding_model_endpoint_name #The embedding endpoint used to create the embeddings
  )
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  wait_for_index_to_be_ready(vsc, vector_search_endpoint_name, vector_search_index)
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, vector_search_endpoint_name, vector_search_index)
  vsc.get_index(vector_search_endpoint_name, vector_search_index).sync()

print(f"index {vector_search_index} on table {source_table_name} is ready")

# COMMAND ----------

from typing import Dict, Any, List

import mlflow.deployments
from databricks.vector_search.index import VectorSearchIndex
from mlflow.deployments.databricks import DatabricksDeploymentClient

deploy_client: DatabricksDeploymentClient = mlflow.deployments.get_deploy_client("databricks")

question = "How can add a fundraiser key?"

index: VectorSearchIndex = vsc.get_index(vector_search_endpoint_name, vector_search_index)

document_uri: str = "source"
search_results: Dict[str, Any] = index.similarity_search(
  query_text=question,
  columns=[document_uri, "content"],
  num_results=3)

chunks: List[str] = search_results.get('result', {}).get('data_array', [])
chunks

# COMMAND ----------

from typing import Dict, Any

import os

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex


vsc: VectorSearchClient = VectorSearchClient(
    workspace_url=workspace_url,
    service_principal_client_id=client_id,
    service_principal_client_secret=client_secret
)

# vsc: VectorSearchClient = VectorSearchClient(
#     workspace_url=workspace_url,
#     personal_access_token=os.environ.get("DATABRICKS_TOKEN")
# )

index: VectorSearchIndex = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vector_search_index)

search_results: Dict[str, Any] = (
    index.similarity_search(num_results=3, columns=["source", "content"], query_text=question)
)

search_results
                

# COMMAND ----------

print(workspace_url)

# COMMAND ----------

from typing import Any, Dict, List, Optional

from langchain_core.vectorstores.base import VectorStoreRetriever
from databricks_langchain.vectorstores import DatabricksVectorSearch

import mlflow.models


from langchain.schema import Document
from typing import List
import os

def create_vector_search_retriever(
    endpoint_name: str,
    index_name: str,
    primary_key: Optional[str] = None,
    text_column: Optional[str] = None,
    doc_uri: Optional[str] = None,
    columns: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> VectorStoreRetriever:


    print("@@@@@@@@client_id", client_id)
    print("@@@@@@@@client_secret", client_secret)
    if client_id and client_secret:
        os.environ["DATABRICKS_CLIENT_ID"] = client_id
        os.environ["DATABRICKS_CLIENT_SECRET"] = client_secret

    vector_search: DatabricksVectorSearch = DatabricksVectorSearch(
        index_name=index_name,
        endpoint=endpoint_name,
        text_column=text_column,
        columns=columns,
        client_args={
            "service_principal_client_id": client_id,
            "service_principal_client_secret": client_secret,
        }
    )
    
    vector_search_as_retriever: VectorStoreRetriever = (
        vector_search.as_retriever(search_kwargs=parameters)
    )

    mlflow.models.set_retriever_schema(
        name=index_name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
    )
    
    return vector_search_as_retriever

# COMMAND ----------


