# Databricks notebook source
pip_requirements: str = (
  "langchain "
  "langchain-community "
  "databricks-langchain "
  "databricks-agents "
  "databricks-sdk "
  "mlflow "
  "python-dotenv "
)

%pip install --upgrade --quiet {pip_requirements}
%restart_python

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from typing import Sequence

from importlib.metadata import version

pip_requirements: Sequence[str] = [
    f"langchain=={version('langchain')}",
    f"langchain-community=={version('langchain-community')}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"mlflow=={version('mlflow')}",
]
print(pip_requirements)

# COMMAND ----------

# MAGIC %%writefile agent_as_code.py
# MAGIC
# MAGIC from typing import List, Dict, Any, Iterator, Sequence, Union
# MAGIC
# MAGIC import mlflow
# MAGIC from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, messages_from_dict
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC # Model definitions
# MAGIC supervisor_model_name = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC chat_model = ChatDatabricks(endpoint=supervisor_model_name)
# MAGIC
# MAGIC tableau_model_name = "agents_partner_summary-engagements-engagement_summary_bot"
# MAGIC tableau_model = ChatDatabricks(endpoint=tableau_model_name)
# MAGIC
# MAGIC data_discovery_model_name = "agents_partner_summary-engagements-engagement_summary_bot"
# MAGIC data_discovery_model = ChatDatabricks(endpoint=data_discovery_model_name)
# MAGIC
# MAGIC # Router prompt template
# MAGIC router_prompt = """
# MAGIC You are a router that will analyze the user's question and determine which specialized system should handle it.
# MAGIC Route based on these criteria:
# MAGIC - Questions about dashboards and reports should route to "tableau"
# MAGIC - Questions about catalogs, schema and columns should route to "unity_catalog"
# MAGIC - All other questions should route to "general"
# MAGIC
# MAGIC User question: {question}
# MAGIC
# MAGIC Please respond with only one of these exact words: "tableau", "unity_catalog", or "general"
# MAGIC """
# MAGIC
# MAGIC
# MAGIC def extract_last_message(messages: List[BaseMessage]) -> str:
# MAGIC     """Extract the content from the last message"""
# MAGIC     if not messages:
# MAGIC         return ""
# MAGIC     return messages[-1].content
# MAGIC
# MAGIC def determine_route(messages: List[BaseMessage]) -> str:
# MAGIC     """Determine which route to take based on the message content"""
# MAGIC     question = extract_last_message(messages)
# MAGIC     # Construct message for the router
# MAGIC     router_messages = [
# MAGIC         HumanMessage(content=router_prompt.format(question=question))
# MAGIC     ]
# MAGIC     # Get routing decision
# MAGIC     response = chat_model.invoke(router_messages)
# MAGIC     # Extract and normalize the routing decision
# MAGIC     route = response.content.strip().lower()
# MAGIC     # Ensure valid route
# MAGIC     if route not in ["tableau", "unity_catalog", "general"]:
# MAGIC         route = "general"
# MAGIC     return route
# MAGIC
# MAGIC def handle_tableau_question(messages: List[BaseMessage]) -> AIMessage:
# MAGIC     print("handle_tableau_question")
# MAGIC     return tableau_model.invoke(messages)
# MAGIC
# MAGIC def handle_unity_catalog_question(messages: List[BaseMessage]) -> AIMessage:
# MAGIC     print("handle_unity_catalog_question")
# MAGIC     return data_discovery_model.invoke(messages)
# MAGIC
# MAGIC def handle_general_question(messages: List[BaseMessage]) -> AIMessage:
# MAGIC     print("handle_general_question")
# MAGIC     return AIMessage(content="I am unable to route that question")
# MAGIC
# MAGIC
# MAGIC def route_and_process(messages: List[BaseMessage]) -> AIMessage:
# MAGIC     print("route_and_process")
# MAGIC
# MAGIC     route = determine_route(messages)
# MAGIC
# MAGIC     if route == "tableau":
# MAGIC         return handle_tableau_question(messages)
# MAGIC     elif route == "unity_catalog":
# MAGIC         return handle_unity_catalog_question(messages)
# MAGIC     else:
# MAGIC         return handle_general_question(messages)
# MAGIC
# MAGIC
# MAGIC def parse_messages(payload: Union[Dict[str, Any], List[BaseMessage]]) -> List[BaseMessage]:
# MAGIC
# MAGIC     if isinstance(payload, list) and all(isinstance(msg, BaseMessage) for msg in payload):
# MAGIC         return payload
# MAGIC
# MAGIC     if isinstance(payload, dict) and "messages" in payload:
# MAGIC         return messages_from_dict(payload["messages"])
# MAGIC     
# MAGIC     if isinstance(payload, list) and all(isinstance(msg, dict) for msg in payload):
# MAGIC         return messages_from_dict(payload)
# MAGIC     
# MAGIC     if isinstance(payload, dict):
# MAGIC         return messages_from_dict([payload])
# MAGIC
# MAGIC
# MAGIC def handle_tableau_question_stream(messages: List[BaseMessage]) -> Iterator[AIMessage]:
# MAGIC     """Handle tableau-related questions with streaming"""
# MAGIC     for chunk in tableau_model.stream(messages):
# MAGIC         yield chunk
# MAGIC
# MAGIC def handle_unity_catalog_question_stream(messages: List[BaseMessage]) -> Iterator[AIMessage]:
# MAGIC     """Handle unity catalog-related questions with streaming"""
# MAGIC     for chunk in data_discovery_model.stream(messages):
# MAGIC         yield chunk
# MAGIC
# MAGIC def handle_general_question_stream(messages: List[BaseMessage]) -> Iterator[AIMessage]:
# MAGIC     """Handle general questions with streaming"""
# MAGIC     yield AIMessage(content="I am unable to route that question")
# MAGIC
# MAGIC
# MAGIC def route_and_process_stream(messages: List[BaseMessage]) -> Iterator[AIMessage]:
# MAGIC     """Route the question and process it with the appropriate streaming handler"""
# MAGIC     # First determine the route
# MAGIC     route = determine_route(messages)
# MAGIC     
# MAGIC     # Then use the appropriate streaming handler
# MAGIC     if route == "tableau":
# MAGIC         for chunk in handle_tableau_question_stream(messages):
# MAGIC             yield chunk
# MAGIC     elif route == "unity_catalog":
# MAGIC         for chunk in handle_unity_catalog_question_stream(messages):
# MAGIC             yield chunk
# MAGIC     else:
# MAGIC         for chunk in handle_general_question_stream(messages):
# MAGIC             yield chunk
# MAGIC
# MAGIC # Create chains
# MAGIC def create_routing_chain():
# MAGIC     return RunnableLambda(parse_messages) | RunnableLambda(route_and_process)
# MAGIC
# MAGIC def create_streaming_routing_chain():
# MAGIC     return RunnableLambda(parse_messages) | RunnableLambda(route_and_process_stream)
# MAGIC
# MAGIC # Main function to handle a conversation turn
# MAGIC def process_messages(messages: List[BaseMessage]) -> AIMessage:
# MAGIC     chain = create_routing_chain()
# MAGIC     response = chain.invoke(messages)
# MAGIC     return response
# MAGIC
# MAGIC # Streaming version of process_messages
# MAGIC def process_messages_stream(messages: List[BaseMessage]) -> Iterator[AIMessage]:
# MAGIC     chain = create_streaming_routing_chain()
# MAGIC     for chunk in chain.stream(messages):
# MAGIC         yield chunk
# MAGIC
# MAGIC
# MAGIC chain = create_streaming_routing_chain()
# MAGIC
# MAGIC mlflow.models.set_model(chain)
# MAGIC

# COMMAND ----------

import mlflow 

from langchain_core.messages import HumanMessage

messages = [HumanMessage(content="Show me the sales dashboard")]
response = process_messages(messages)
response

# COMMAND ----------

response

# COMMAND ----------

import mlflow
from mlflow.types import DataType, Schema, ColSpec
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest

# Create input schema (single string input)
input_schema = Schema([
    ColSpec(type=DataType.string, name="input")
])

# Create output schema (single string output)
output_schema = Schema([
    ColSpec(type=DataType.string, name="output")
])

# Create the model signature
model_signature = mlflow.models.ModelSignature(
    inputs=ChatCompletionRequest,
    outputs=StringResponse
)

model_signature

# COMMAND ----------

from typing import Sequence

import mlflow
from mlflow.models.model import ModelInfo
from mlflow.models.resources import DatabricksResource, DatabricksServingEndpoint
from mlflow.models.signature import ModelSignature


from agent_as_code import (
  supervisor_model_name, 
  tableau_model_name, 
  data_discovery_model_name
)


input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is Databricks?",
        }
    ]
}


resources: Sequence[DatabricksResource] = [
  DatabricksServingEndpoint(endpoint_name=supervisor_model_name),
  DatabricksServingEndpoint(endpoint_name=tableau_model_name),
  DatabricksServingEndpoint(endpoint_name=data_discovery_model_name),
]

with mlflow.start_run():
    logged_agent_info: ModelInfo = mlflow.langchain.log_model(
        lc_model="agent_as_code.py",
        artifact_path="agent",
        pip_requirements=pip_requirements,
        resources=resources,
        input_example=input_example,
        signature=model_signature
    )

# COMMAND ----------

import mlflow

mlflow.models.predict(
    model_uri=logged_agent_info.model_uri,
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
)

# COMMAND ----------

import mlflow 
from mlflow.entities.model_registry.model_version import ModelVersion


mlflow.set_registry_uri("databricks-uc")

catalog:str = "nfleming"
schema: str = "default"
model_name: str = "delphi-supervisor"
registered_model_name: str = f"{catalog}.{schema}.{model_name}"

# register the model to UC
registred_model_info: ModelVersion = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=registered_model_name
)

# COMMAND ----------

from databricks import agents

agents.deploy(
  registered_model_name, 
  registred_model_info.version, 
  tags={
    "client": "Nike",
    "project": "Delphi",
    "use_case": "POC",
  }
)