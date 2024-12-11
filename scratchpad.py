# Databricks notebook source
# MAGIC %pip install --upgrade langchain mlflow 
# MAGIC %pip install langgraph langchain-openai databricks-langchain langchain-community databricks-vectorsearch databricks-sdk python-dotenv guardrails-ai presidio-analyzer nltk
# MAGIC %restart_python

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
  f"guardrails-ai=={version('guardrails-ai')}",
  f"nltk=={version('nltk')}",
  f"presidio-analyzer=={version('presidio-analyzer')}",
]
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

model_config_file: str = "model_config.yaml"

# COMMAND ----------

import app.llms
import app.graph
import app.messages
import app.agents
import app.prompts
import app.router
import app.tools
import app.guardrails.validators
import app.guardrails.guards

# COMMAND ----------

from typing import Any, Dict, Optional
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSequence
from langgraph.graph import StateGraph

import mlflow
from mlflow.models import ModelConfig

from app.llms import get_llm
from app.graph import GraphBuilder
from app.agents import (
  Agent, 
  create_genie_agent,
  create_vector_search_agent,
  create_unity_catalog_agent,
)

model_config_file: str = "model_config.yaml"
model_config: ModelConfig = ModelConfig(development_config=model_config_file)

databricks_resources: Dict[str, Any] = model_config.get("databricks_resources")
retriever_config: Dict[str, Any] = model_config.get("retriever_config")

space_id: str = databricks_resources.get("genie_space_id")
genie_workspace_host: Optional[str] = databricks_resources.get("genie_workspace_host")

llm_endpoint_name: str = databricks_resources.get("llm_endpoint_name")

llm: BaseChatModel = get_llm(llm_endpoint_name)


unity_catalog_agent: Agent = create_unity_catalog_agent(
  llm=llm,
  warehouse_id=databricks_resources.get("warehouse_id"),
  functions=[databricks_resources.get("unity_catalog_functions")],
)

genie_agent: Agent = create_genie_agent(
    llm=llm,
    space_id=space_id,
)

vector_search_schema: Dict[str, str] = retriever_config.get("schema")

vector_search_agent: Agent = create_vector_search_agent(
  llm=llm,
  endpoint_name = databricks_resources.get("vector_search_endpoint_name"),
  index_name = retriever_config.get("vector_search_index"),
  columns = [
      vector_search_schema.get("primary_key"),
      vector_search_schema.get("chunk_text"),
      vector_search_schema.get("document_uri"),
  ],
  parameters = retriever_config.get("parameters"),
)

builder: GraphBuilder = (
  GraphBuilder(llm=llm)
    .add_agent(vector_search_agent)
    .add_agent(genie_agent)
    .add_agent(unity_catalog_agent)
    .with_debug()
    #.with_memory()
)

graph: StateGraph = builder.build()

chain: RunnableSequence = graph.as_chain()

mlflow.models.set_model(chain)

# COMMAND ----------

from typing import List

from langchain_core.messages import HumanMessage

mlflow.langchain.autolog(disable=False)

message: str = "How can I optimize a Databricks Cluster?"
messages: List[HumanMessage] = [HumanMessage(content=message)]
config: Dict[str, Any] = {
    "configurable": {"thread_id": 42}
}

response: Dict[str, Any] = chain.invoke(messages, config=config)
response


# COMMAND ----------

from typing import List

from langchain_core.messages import HumanMessage


message: str = "How many rows of documentation are there in the genie space?"
messages: List[HumanMessage] = [HumanMessage(content=message)]
config: Dict[str, Any] = {
    "configurable": {"thread_id": 42}
}

input_example = {
        "messages": [
            {
                "role": "user",
                "content": message,
            }
        ]
    }

# Use the Runnable
final_state = chain.invoke(
    #input_example,
    #{"messages": messages},
    messages[0]

   # config=config,
)

# final_state: List[BaseMessage] = graph.invoke(messages, config=config)
# final_state[-1].content



# COMMAND ----------

graph.display(verbose=False)

# COMMAND ----------

from typing import List
from langchain_core.messages import HumanMessage

messages: List[HumanMessage] = [
  HumanMessage(content="What is 9 * 49?")
]
print(unity_catalog_agent.as_runnable().invoke(messages))

# COMMAND ----------

from typing import List
from langchain_core.messages import HumanMessage

messages: List[HumanMessage] = [HumanMessage(content="What is Databricks?")]
print(vector_search_agent.as_runnable().invoke(messages))

# COMMAND ----------

from typing import List
from langchain_core.messages import HumanMessage

messages: List[HumanMessage] = [
  HumanMessage(content="How many rows of documentation are there in the Genie space?")
]
print(genie_agent.as_runnable().invoke(messages))

# COMMAND ----------


genie_tool = genie_agent.tools[0]
print(type(genie_tool))
genie_tool.run("How many rows of documentation are there in the genie space?")

# COMMAND ----------

from typing import List

from langchain_core.messages import HumanMessage


message: str = "Use the UC functions to compute 49 * 9?"
messages: List[HumanMessage] = [HumanMessage(content=message)]
config: Dict[str, Any] = {
    "configurable": {"thread_id": 42}
}

response: Dict[str, Any] = chain.invoke(messages, config=config)
response

