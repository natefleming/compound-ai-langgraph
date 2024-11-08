# Databricks notebook source
# MAGIC %pip install --quiet --upgrade langchain langgraph langchain-openai langchain-databricks langchain-community mlflow databricks-sdk databricks-vectorsearch databricks-agents python-dotenv nest-asyncio rich
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from rich import print

# COMMAND ----------

from typing import List

from importlib.metadata import version

pip_requirements: List[str] = [
  f"mlflow=={version('mlflow')}",
  f"langchain=={version('langchain')}",
  f"langchain_openai=={version('langchain_openai')}",
  f"langchain_databricks=={version('langchain_databricks')}",
  f"langchain_community=={version('langchain_community')}",
  f"langgraph=={version('langgraph')}",
  f"databricks_sdk=={version('databricks_sdk')}",
  f"databricks_vectorsearch=={version('databricks_vectorsearch')}",
  f"nest-asyncio=={version('nest-asyncio')}",
]
pip_requirements = [f"\"{p}\"," for p in pip_requirements]
print("\n".join(pip_requirements))

# COMMAND ----------

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import os 
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host: str = spark.conf.get("spark.databricks.workspaceUrl")
base_url: str = f"{workspace_host}/serving-endpoints/"
token: str = get_databricks_host_creds().token

os.environ["DATABRICKS_HOST"] = f"https://{workspace_host}"
os.environ["DATABRICKS_TOKEN"] = token

print(f"workspace_host: {workspace_host}")
print(f"base_url: {base_url}")

# COMMAND ----------

from importlib import reload

import app.tools
import app.graph
import app.messages
import app.agents
import app.prompts
import app.router
import app.genie_utils

reload(app.tools)
reload(app.llms)
reload(app.graph)
reload(app.messages)
reload(app.agents)
reload(app.prompts)
reload(app.router)
reload(app.genie_utils)


# COMMAND ----------

from pathlib import Path

import mlflow

from mlflow.tracking import MlflowClient
from mlflow.models.model import ModelInfo
from mlflow.models.signature import ModelSignature, ParamSchema
from mlflow.types import DataType, ParamSpec
from mlflow.models.rag_signatures import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

mlflow.set_registry_uri("databricks-uc")

registered_model_name: str = "nfleming.default.compound_ai_langgraph"

signature: ModelSignature = ModelSignature(
    inputs=ChatCompletionRequest(),
    outputs=ChatCompletionResponse(),
    # params=ParamSchema([
    #     ParamSpec("temperature", DataType.float, 0.1, None),
    #     ParamSpec("max_tokens", DataType.integer, 1000, None),
    #     ParamSpec("thread_id", DataType.integer, 42, None),
    # ])
)

with mlflow.start_run(run_name="chain"):
 
    mlflow.set_tag("type", "chain")
    model_info: ModelInfo = mlflow.langchain.log_model(
        lc_model="chain_as_code.py",
        registered_model_name=registered_model_name,
        code_paths=["app"],
        model_config="model_config.yaml",
        artifact_path="chain",
        signature=signature,
        example_no_conversion=True,
        pip_requirements=[
            "mlflow==2.17.1",
            "langchain==0.3.5",
            "langchain_openai==0.2.4",
            "langchain_databricks==0.1.1",
            "langchain_community==0.3.3",
            "langgraph==0.2.39",
            "databricks_sdk==0.36.0",
            "databricks_vectorsearch==0.40",
            "nest-asyncio==1.6.0",
        ],
    )
    print(model_info.registered_model_version)


# COMMAND ----------

def get_latest_model_version(model_name: str) -> int:
    from mlflow.tracking import MlflowClient
    mlflow_client: MlflowClient = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

latest_model_version: int = get_latest_model_version(registered_model_name)

print(f"{latest_model_version=}")

# COMMAND ----------

from mlflow.tracking import MlflowClient

client: MlflowClient = MlflowClient(registry_uri="databricks-uc")

client.set_registered_model_alias(
  name=registered_model_name, 
  alias="Champion", 
  version=latest_model_version
)

# COMMAND ----------

from mlflow.pyfunc import PyFuncModel

loaded_model: PyFuncModel = (
  mlflow.pyfunc.load_model(f"models:/{registered_model_name}@Champion")
)

# COMMAND ----------

from typing import Dict


payload: Dict[str, str] = {
    "messages": [
        {
            "role": "user",
            "content": "How many rows of documentation are there in the genie space?",
            #"content": "How can we optimize clusters in Databricks?",
        }
    ]
}

config: Dict[str, str] = {
    "thread_id": 42
}

response = loaded_model.predict(payload)

# COMMAND ----------

import os
import time

from mlflow.utils import databricks_utils as du


from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist, NotFound, PermissionDenied
from databricks.agents import deploy
from databricks.agents.sdk_utils.entities import Deployment


w: WorkspaceClient = WorkspaceClient()


deployment: Deployment = deploy(
    registered_model_name, 
    latest_model_version,
    environment_vars={
       # "DATABRICKS_TOKEN": f"{{secrets/{scope}/databricks_token}}", 
       # "DATABRICKS_HOST": f"{{secrets/{scope}/databricks_host}}", 
        "DATABRICKS_HOST": os.environ["DATABRICKS_HOST"], 
        "DATABRICKS_TOKEN": os.environ["DATABRICKS_TOKEN"], 
    }
)

print(f"endpoint_name: {deployment.endpoint_name}")
print(f"endpoint_url: {deployment.endpoint_url}")
print(f"query_endpoint: {deployment.query_endpoint}")
print(f"review_app_url: {deployment.review_app_url}")

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 10 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(60)

# COMMAND ----------

from typing import Dict 

import json
import requests


payload: Dict[str, str] = {
    "messages": [
        {
            "role": "user",
            #"content": "How many rows of documentation are there in the genie space?",
            "content": "How can we optimize clusters in Databricks?",
        }
    ]
}

url: str = f"https://{workspace_host}/serving-endpoints/agents_nfleming-default-compound_ai_langgraph/invocations"
headers: Dict[str, str] = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
response = requests.request(method='POST', headers=headers, url=url, json=payload)
if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

print(response.content)

# COMMAND ----------

from typing import List
import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole, QueryEndpointResponse, V1ResponseChoiceElement


w: WorkspaceClient = WorkspaceClient()

def ask(message: str) -> str:
    messages: List[ChatMessage] = []

    messages.append(ChatMessage(content=message, role=ChatMessageRole.USER))
    print(messages)
    response: QueryEndpointResponse = w.serving_endpoints.query(
        name="agents_nfleming-default-compound_ai_langgraph",
        messages=messages,
        temperature=1.0,
        stream=False,
    )

    choices: List[V1ResponseChoiceElement] = response.choices
    respone_message: ChatMessage = choices[0].message
    response_content: str = respone_message.content
    return response_content

# COMMAND ----------

messages: List[ChatMessage] = []

message: str = "How many rows of documentation are there in the genie space?"
messages.append(ChatMessage(content=message, role=ChatMessageRole.USER))
response: QueryEndpointResponse = w.serving_endpoints.query(
    name="agents_nfleming-default-compound_ai_langgraph",
    messages=messages,
    temperature=1.0,
    stream=False,
)
response


# COMMAND ----------

response: str = ask("How many rows of documentation are there in the genie space?")
print(response)

print("-" * 80)

response: str = ask("How can you optimize a Databricks Cluster?")
print(response)

print("-" * 80)

response: str = ask("What is the capitol of Indiana?")
print(response)
