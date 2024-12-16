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
  create_vector_search_chain,
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


genie_agent: Agent = create_genie_agent(
    llm=llm,
    space_id=space_id,
)

vector_search_schema: Dict[str, str] = retriever_config.get("schema")
    
vector_search_agent: Agent = create_vector_search_agent(
  llm=llm,
  endpoint_name = databricks_resources.get("vector_search_endpoint_name"),
  index_name = retriever_config.get("vector_search_index"),
  primary_key = vector_search_schema.get("primary_key"),
  text_column = vector_search_schema.get("chunk_text"),
  doc_uri = vector_search_schema.get("document_uri"),
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
    .default_agent(vector_search_agent)
    #.with_debug()
    #.with_memory()
)

graph: StateGraph = builder.build()

chain: RunnableSequence = graph.as_chain()

mlflow.models.set_model(chain)