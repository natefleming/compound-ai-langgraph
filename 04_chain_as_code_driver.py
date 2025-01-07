# Databricks notebook source
# MAGIC %pip install --quiet --upgrade langchain langgraph langchain-openai databricks-langchain langchain-community mlflow databricks-sdk databricks-vectorsearch databricks-agents python-dotenv #guardrails-ai presidio-analyzer nltk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from typing import List

from importlib.metadata import version

pip_requirements: List[str] = [
  f"mlflow=={version('mlflow')}",
  f"langchain=={version('langchain')}",
  f"langchain_openai=={version('langchain_openai')}",
  f"databricks_langchain=={version('databricks_langchain')}",
  f"langchain_community=={version('langchain_community')}",
  f"langgraph=={version('langgraph')}",
  f"databricks_sdk=={version('databricks_sdk')}",
  f"databricks_vectorsearch=={version('databricks_vectorsearch')}",
  # f"guardrails-ai=={version('guardrails-ai')}",
  # f"nltk=={version('nltk')}",
  # f"presidio-analyzer=={version('presidio-analyzer')}",
]
pip_requirements = [f"\"{p}\"," for p in pip_requirements]
print("\n".join(pip_requirements))

# COMMAND ----------

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import app.llms
import app.graph
import app.messages
import app.agents
import app.prompts
import app.router
import app.retrievers
import app.tools
import app.catalog
# import app.guardrails.validators
# import app.guardrails.guards

# COMMAND ----------

from typing import Any, Dict
from mlflow.models import ModelConfig

model_config_file: str = "model_config.yaml"
model_config: ModelConfig = ModelConfig(development_config=model_config_file)

databricks_resources: Dict[str, Any] = model_config.get("databricks_resources")
registered_model_name: str = databricks_resources.get("registered_model_name")
evaluation_table_name: str = databricks_resources.get("evaluation_table_name")
curated_evaluation_table_name: str = databricks_resources.get("curated_evaluation_table_name")

secret_scope: str = databricks_resources.get("scope_name")
secret_name: str = databricks_resources.get("secret_name")
client_id: str = databricks_resources.get("client_id")
client_secret: str = databricks_resources.get("client_secret")

print(f"{registered_model_name=}") 
print(f"{evaluation_table_name=}")
print(f"{curated_evaluation_table_name=}")
print(f"{secret_scope=}")
print(f"{secret_name=}")
print(f"{client_id=}")
print(f"{client_secret=}")

# COMMAND ----------

import os 
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host: str = spark.conf.get("spark.databricks.workspaceUrl")
base_url: str = f"{workspace_host}/serving-endpoints/"
#token: str = get_databricks_host_creds().token

os.environ["DATABRICKS_HOST"] = f"https://{workspace_host}"
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(secret_scope, secret_name)


print(f"workspace_host: {workspace_host}")
print(f"base_url: {base_url}")

# COMMAND ----------

from typing import Union, Tuple, Optional
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
from mlflow.models.evaluation import EvaluationResult

import pandas as pd


mlflow.set_registry_uri("databricks-uc")
 
signature: ModelSignature = ModelSignature(
    inputs=ChatCompletionRequest(),
    outputs=ChatCompletionResponse(),
)

def log_and_evaluate_agent(
    run_name: str, 
    model_config: Union[ModelConfig, Dict[str, Any]], 
    eval_df: Optional[pd.DataFrame] = None,
) -> Tuple[ModelInfo, EvaluationResult]:

    if isinstance(model_config, ModelConfig):
        model_config = model_config.to_dict()

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("type", "chain")
        model_info: ModelInfo = mlflow.langchain.log_model(
            lc_model="chain_as_code.py",
            registered_model_name=registered_model_name,
            code_paths=["app"],
            model_config=model_config,
            artifact_path="chain",
            signature=signature,
            example_no_conversion=True,
            pip_requirements=[
                "mlflow==2.19.0",
                "langchain==0.3.13",
                "langchain_openai==0.2.13",
                "databricks_langchain==0.1.1",
                "langchain_community==0.3.13",
                "langgraph==0.2.60",
                "databricks_sdk==0.40.0",
                "databricks_vectorsearch==0.43",
                # "guardrails-ai==0.6.1",
                # "nltk==3.9.1",
                # "presidio-analyzer==2.2.356",
            ],
        )

        print(model_info.registered_model_version)

        eval_results: Optional[EvaluationResult] = None
        if eval_df is not None:
            eval_results = mlflow.evaluate(
                data=eval_df,             # Your evaluation set
                model=model_info.model_uri,     # Logged agent from above
                model_type="databricks-agent",  # activate Mosaic AI Agent Evaluation
            )
        return (model_info, eval_results)


# COMMAND ----------

from mlflow.models.model import ModelInfo
from mlflow.models.evaluation import EvaluationResult

model_info: mlflow.models.model.ModelInfo
evaluation_result: EvaluationResult

evalution_pdf: pd.DataFrame = spark.table(evaluation_table_name).toPandas()
#currated_evaluation_pdf: pd.DataFrame = spark.table(curated_evaluation_table_name).toPandas()
#currated_evaluation_pdf = None

model_info, evaluation_result = (
  log_and_evaluate_agent(run_name="chain", model_config=model_config, eval_df=evalution_pdf)
)


# COMMAND ----------

from mlflow.tracking import MlflowClient

from app.utils import get_latest_model_version

client: MlflowClient = MlflowClient(registry_uri="databricks-uc")
latest_model_version: int = get_latest_model_version(registered_model_name)

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
            "content": "Where can I order a printer for receipts?",
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


from databricks.sdk.service.serving import (
    EndpointStateReady, 
    EndpointStateConfigUpdate, 
    ServedModelInputWorkloadSize
)
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist, NotFound, PermissionDenied
from databricks.agents import deploy
from databricks.agents.sdk_utils.entities import Deployment


w: WorkspaceClient = WorkspaceClient()


deployment: Deployment = deploy(
    registered_model_name, 
    latest_model_version,
    scale_to_zero=True,
    workload_size = ServedModelInputWorkloadSize.MEDIUM,
    environment_vars={
        #"DATABRICKS_TOKEN": f"{{secrets/{secret_scope}/{secret_name}}}", 
        #"DATABRICKS_HOST": f"{{secrets/{secret_scope}/{databricks_host}}}", 
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
