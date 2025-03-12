# Databricks notebook source
# MAGIC %pip install --quiet --upgrade langchain langgraph langchain-community databricks-langchain mlflow trustcall python-dotenv
# MAGIC %restart_python

# COMMAND ----------

llm: BaseChatModel = ChatDatabricks(endpoint="agents_nfleming-default-langgraph_chatagent")

# COMMAND ----------

from importlib.metadata import version

from typing import Sequence

from importlib.metadata import version

pip_requirements: Sequence[str] = [
    f"langchain=={version('langchain')}",
    f"langgraph=={version('langgraph')}",
    f"langchain-community=={version('langchain-community')}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"mlflow=={version('mlflow')}",
    f"trustcall=={version('trustcall')}",
]
print("\n".join(pip_requirements))


# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %%writefile agent_as_code.py
# MAGIC from typing import (
# MAGIC     Sequence, 
# MAGIC     Annotated, 
# MAGIC     TypedDict,
# MAGIC     Annotated, 
# MAGIC     Literal, 
# MAGIC     Optional, 
# MAGIC     Any, 
# MAGIC     Generator
# MAGIC )
# MAGIC
# MAGIC from langchain.prompts import PromptTemplate
# MAGIC from langchain_core.language_models.chat_models import BaseChatModel
# MAGIC from langchain_core.messages import AIMessage, BaseMessage, ChatMessageChunk
# MAGIC from langchain_core.runnables import RunnableSequence
# MAGIC from langchain_core.vectorstores.base import VectorStore
# MAGIC from langchain_core.documents.base import Document
# MAGIC
# MAGIC from langgraph.graph import StateGraph, END
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC from databricks_langchain import DatabricksVectorSearch
# MAGIC from databricks_langchain.genie import Genie
# MAGIC from databricks_ai_bridge.genie import GenieResponse
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC from pydantic import BaseModel, Field
# MAGIC
# MAGIC from trustcall import create_extractor
# MAGIC
# MAGIC
# MAGIC allowed_routes: Sequence[str] = ["code", "general", "genie", "tableau"]
# MAGIC
# MAGIC class Router(BaseModel):
# MAGIC   """
# MAGIC   A router that will route the question to the right agent. 
# MAGIC   * Questions about finance should route to genie. 
# MAGIC   * Questions about marketing should route to tableau.
# MAGIC   * Questions about code should route to code.
# MAGIC   * All other questions should route to general
# MAGIC   """
# MAGIC   route: Literal[tuple(allowed_routes)] = Field(default="general", description=f"The route to follow should be one of: {', '.join(allowed_routes)}")
# MAGIC
# MAGIC
# MAGIC class AgentState(ChatAgentState):
# MAGIC   messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC   context: Sequence[Document]
# MAGIC   route: str
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def route_question(state: AgentState) -> dict[str, str]:
# MAGIC     print("route_question")
# MAGIC     llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
# MAGIC     chain: RunnableSequence = llm.with_structured_output(Router)
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     last_message: BaseMessage = messages[-1]
# MAGIC     response = chain.invoke([last_message])
# MAGIC     print(f"route: {response}" )
# MAGIC     return {"route": response.route}
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def vector_search_question(state: AgentState) -> dict[str, str]:
# MAGIC     print(vector_search_question)
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = messages[-1].content
# MAGIC
# MAGIC     vector_search: VectorStore = DatabricksVectorSearch(
# MAGIC         endpoint=endpoint,
# MAGIC         index_name=index_name,
# MAGIC     )
# MAGIC
# MAGIC     context: Sequence[Document] = vector_search.similarity_search(
# MAGIC         query=content, k=1, filter={}
# MAGIC     )
# MAGIC
# MAGIC     return {"context": context}
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def genie_question(state: AgentState) -> dict[str, str]:
# MAGIC     print(genie_question)
# MAGIC     print(state)
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = messages[-1].content
# MAGIC
# MAGIC     space_id = "01efeed5e74c1c718b07e0662f90e06e"
# MAGIC     genie: Genie = Genie(space_id=space_id)
# MAGIC     genie_response: GenieResponse = genie.ask_question(content)
# MAGIC     description: str = genie_response.description
# MAGIC     result: str = genie_response.result
# MAGIC     response: str = f"{description}\n{result}"
# MAGIC     return {"messages": [response]}
# MAGIC   
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def code_question(state: AgentState) -> dict[str, BaseMessage]:
# MAGIC     llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
# MAGIC     prompt: PromptTemplate = PromptTemplate.from_template(
# MAGIC         "You are a software engineer. Answer this question with step by steps details : {input}"
# MAGIC     )
# MAGIC     chain: RunnableSequence = prompt | llm
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = messages[-1].content
# MAGIC     response = chain.invoke({"input": content})
# MAGIC     return {"messages": [response]}
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def tableau_question(state: AgentState) -> dict[str, BaseMessage]:
# MAGIC     llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
# MAGIC     prompt: PromptTemplate = PromptTemplate.from_template(
# MAGIC         "You are a Business Analyst. Answer this question with step by steps details : {input}"
# MAGIC     )
# MAGIC     chain: RunnableSequence = prompt | llm
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = messages[-1].content
# MAGIC     response = chain.invoke({"input": content})
# MAGIC     return {"messages": [response]}
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def generic_question(state: AgentState) -> dict[str, BaseMessage]:
# MAGIC     llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
# MAGIC     prompt: PromptTemplate = PromptTemplate.from_template(
# MAGIC         "Give a general and concise answer to the question: {input}"
# MAGIC     )
# MAGIC     chain: RunnableSequence = prompt | llm
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = messages[-1].content
# MAGIC     response = chain.invoke({"input": content})
# MAGIC     return {"messages": [response]}
# MAGIC   
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def summarize_response(state: AgentState) -> dict[str, BaseMessage]:
# MAGIC     llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
# MAGIC     prompt: PromptTemplate = PromptTemplate.from_template(
# MAGIC         "Summarize: {input}"
# MAGIC     )
# MAGIC     chain: RunnableSequence = prompt | llm
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     response = chain.invoke(messages)
# MAGIC     return {"messages": [response]}
# MAGIC
# MAGIC
# MAGIC def create_graph() -> CompiledStateGraph:
# MAGIC     workflow: StateGraph = StateGraph(AgentState)
# MAGIC
# MAGIC     workflow.add_node("router", route_question)
# MAGIC     workflow.add_node("code_agent", code_question)
# MAGIC     workflow.add_node("generic_agent", generic_question)
# MAGIC     workflow.add_node("genie_agent", genie_question)
# MAGIC     workflow.add_node("tableau_agent", tableau_question)
# MAGIC     workflow.add_node("summarize_agent", summarize_response)
# MAGIC
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "router",
# MAGIC         lambda state: state["route"],
# MAGIC         {
# MAGIC             "code": "code_agent",
# MAGIC             "general": "generic_agent",
# MAGIC             "genie": "genie_agent",
# MAGIC             "tableau": "tableau_agent",
# MAGIC         }
# MAGIC     )
# MAGIC
# MAGIC     workflow.set_entry_point("router")
# MAGIC     workflow.add_edge("code_agent", "summarize_agent")
# MAGIC     workflow.add_edge("generic_agent", "summarize_agent")
# MAGIC     workflow.add_edge("genie_agent", "summarize_agent")
# MAGIC     workflow.add_edge("tableau_agent", "summarize_agent")
# MAGIC     workflow.set_finish_point("summarize_agent")
# MAGIC  
# MAGIC     
# MAGIC     return workflow.compile()
# MAGIC     
# MAGIC
# MAGIC graph: CompiledStateGraph = create_graph()
# MAGIC
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates", config=custom_inputs):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in  node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         for event in self.agent.stream(request, stream_mode="updates", config=custom_inputs):
# MAGIC           for node_data in event.values():
# MAGIC             yield from (
# MAGIC               ChatAgentChunk(**{"delta": msg}) for msg in node_data.get("messages", [])
# MAGIC             )
# MAGIC
# MAGIC         # for msg, _ in self.agent.stream(request, stream_mode="messages", config=custom_inputs):
# MAGIC         #   msg: ChatMessageChunk
# MAGIC         #   if not msg.content:
# MAGIC         #     continue
# MAGIC         #   chat_agent_message: ChatAgentMessage = self._langchain_chunk_to_mlflow_chunk(msg)
# MAGIC         #   chat_agent_chunk: ChatAgentChunk = ChatAgentChunk(delta=chat_agent_message)
# MAGIC         #   yield chat_agent_chunk
# MAGIC
# MAGIC     def _langchain_chunk_to_mlflow_chunk(self, langchain_chunk: ChatMessageChunk) -> ChatAgentMessage:
# MAGIC         role_mapping = {
# MAGIC             HumanMessageChunk: "user",
# MAGIC             AIMessageChunk: "assistant",
# MAGIC             SystemMessageChunk: "system",
# MAGIC             ToolMessageChunk: "tool",
# MAGIC         }
# MAGIC         chunk_class = langchain_chunk.__class__
# MAGIC         role = role_mapping.get(chunk_class)
# MAGIC         if not role:
# MAGIC             raise ValueError(f"Unsupported LangChain message type: {chunk_class.__name__}")
# MAGIC         
# MAGIC         return ChatAgentMessage(role=role, content=langchain_chunk.content)
# MAGIC
# MAGIC
# MAGIC app = LangGraphChatAgent(graph)
# MAGIC mlflow.models.set_model(app)
# MAGIC

# COMMAND ----------

from agent_as_code import graph

from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# COMMAND ----------

from agent_as_code import graph

inputs = {
    "messages": [
        ("user", "show me marketing research for Japan"),
    ]
}
response = graph.invoke(inputs)


response["messages"][-1].content

# COMMAND ----------

from agent_as_code import graph

inputs = {
    "messages": [
        ("user", "show me total bank accounts for each country"),
    ]
}
response = graph.invoke(inputs)


response["messages"][-1].content

# COMMAND ----------

from agent_as_code import app

app.predict({"messages": [{"role": "user", "content": "show me marketing research for Japan"}]})

# COMMAND ----------

import mlflow
from agent_as_code import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

# TODO: Manually include underlying resources if needed. See the TODO in the markdown above for more information.
resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))


with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent_as_code.py",
        pip_requirements=pip_requirements,
        resources=resources,
    )

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "nfleming"
schema = "default"
model_name = "langgraph_chatagent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "docs"})

# COMMAND ----------



# COMMAND ----------

import pprint

inputs = {
    "messages": [
        ("user", "show me total bank accounts for each country"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")



# COMMAND ----------

from databricks_langchain import ChatDatabricks
from databricks_langchain import DatabricksVectorSearch
from databricks_langchain.genie import Genie
from databricks_ai_bridge.genie import GenieResponse
from langchain_core.language_models import LanguageModelLike
from langchain.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

import pandas as pd

space_id = "01efeed5e74c1c718b07e0662f90e06e"

content = "What is the market sentiment for companies in my portfolio in the retail industry?"
genie: Genie = Genie(space_id=space_id)
genie_response: GenieResponse = genie.ask_question(content)
# description: str = genie_response.description
result: str = genie_response.result
# response: str = f"{description}\n{result}"

llm: LanguageModelLike = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

json_parser: SimpleJsonOutputParser = SimpleJsonOutputParser()

prompt: PromptTemplate = PromptTemplate.from_template(
    "Convert the following markdown table to json: {markdown}"
)

chain = prompt | llm | json_parser

json_result = chain.invoke({
  "markdown": result
})

df = pd.DataFrame(json_result)

display(df)


# COMMAND ----------

from pprint import pprint

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex 

endpoint="one-env-shared-endpoint-12"
index_name="nfleming.sgws.resolved_entity_chunked_index"

vector_search = VectorSearchClient()

vector_search_index: VectorSearchIndex = vector_search.get_index(endpoint_name=endpoint, index_name=index_name)

result = vector_search_index.similarity_search(["content"], query_text="We can use your drink machine how do we order it OBO?", num_results=1)

# results = vector_search.similarity_search(
#     query="What is OBO?", k=1, filter={}
# )
pprint(result)

