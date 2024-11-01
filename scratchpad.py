# Databricks notebook source
# MAGIC %pip install --quiet --upgrade langchain langgraph langchain-openai langchain-databricks langchain-community mlflow databricks-sdk databricks-vectorsearch python-dotenv rich nest-asyncio
# MAGIC dbutils.library.restartPython()

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
print("\n".join(pip_requirements))

# COMMAND ----------

import sys

from importlib import reload
from rich import print
from dotenv import load_dotenv, find_dotenv



_ = load_dotenv(find_dotenv())

# COMMAND ----------

vector_search_endpoint_name: str = "dbdemos_vs_endpoint" 
vector_search_index: str = "dbdemos.dbdemos_rag_chatbot.databricks_documentation_vs_index"
model_config_file: str = "model_config.yaml"

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

from typing import Any, Dict

import mlflow
from mlflow.models import ModelConfig

import yaml

rag_chain_config: Dict[str, Any] = {
    "databricks_resources": {
        "llm_endpoint_name": "databricks-meta-llama-3-1-70b-instruct",
        "vector_search_endpoint_name": vector_search_endpoint_name,
        "genie_space_id": "01ef9223c3d617e585962e672b22fd2a",
        "genie_workspace_host": "adb-984752964297111.11.azuredatabricks.net"
    },
    "input_example": {
        "messages": [{"content": "Sample user question", "role": "user"}]
    },
    "llm_config": {
        "llm_parameters": {"max_tokens": 1500, "temperature": 0.01},
        "llm_prompt_template": "You are a trusted AI assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know. Here is the history of the current conversation you are having with your user: {chat_history}. And here is some context which may or may not help you answer the following question: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
        "llm_prompt_template_variables": ["context", "chat_history", "question"],
    },
    "retriever_config": {
        "chunk_template": "Passage: {chunk_text}\n",
        "data_pipeline_tag": "poc",
        "parameters": {"k": 5, "query_type": "ann"},
        "schema": {"chunk_text": "content", "document_uri": "url", "primary_key": "id"},
        "vector_search_index": vector_search_index,
    },
}
try:
    with open(model_config_file, 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    ...



# COMMAND ----------

from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.messages import HumanMessage


def text_to_speech(message: HumanMessage) -> str:
  return HumanMessage(content=message.content)

text_to_speech_chain: RunnableSequence = RunnableLambda(text_to_speech)

# COMMAND ----------

from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.messages import HumanMessage


def speech_to_text(message: HumanMessage) -> str:
  return HumanMessage(content=message.content)

speech_to_text_chain: RunnableSequence = RunnableLambda(speech_to_text)

# COMMAND ----------

from importlib import reload

import app.llms
import app.graph
import app.messages
import app.agents
import app.prompts
import app.router
import app.genie_utils

reload(app.llms)
reload(app.graph)
reload(app.messages)
reload(app.agents)
reload(app.prompts)
reload(app.router)
reload(app.genie_utils)



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
  create_vector_search_agent
)

model_config_file: str = "model_config.yaml"
model_config: ModelConfig = ModelConfig(development_config=model_config_file)

databricks_resources: Dict[str, Any] = model_config.get("databricks_resources")
retriever_config: Dict[str, Any] = model_config.get("retriever_config")

space_id: str = databricks_resources.get("genie_space_id")
genie_workspace_host: Optional[str] = databricks_resources.get("genie_workspace_host")

llm_endpoint_name: str = databricks_resources.get("llm_endpoint_name")

llm: BaseChatModel = get_llm(llm_endpoint_name)

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
    #.with_debug()
    #.with_memory()
)

graph: StateGraph = builder.build()

chain: RunnableSequence = graph.as_chain()

mlflow.models.set_model(chain)

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
                "content": "How many rows of documentation are there in the genie space?",
            }
        ]
    }

# Use the Runnable
final_state = chain.invoke(
    #input_example,
    {"messages": messages},
            # {
            #     "role": "user",
            #     "content": "How many rows of documentation are there in the genie space?",
            # },
   # config=config,
)

# final_state: List[BaseMessage] = graph.invoke(messages, config=config)
# final_state[-1].content



# COMMAND ----------

graph.display(verbose=True)

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


message: str = "How can I optimize clusters in Databricks?"
messages: List[HumanMessage] = [HumanMessage(content=message)]
config: Dict[str, Any] = {
    "configurable": {"thread_id": 42}
}

# final_state: List[BaseMessage] = graph.invoke(messages, config=config)
# final_state[-1].content

response: Dict[str, Any] = chain.invoke(messages, config=config)
response

