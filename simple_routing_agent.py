# Databricks notebook source
# MAGIC %pip install --quiet --upgrade langchain langgraph langchain-community databricks-langchain mlflow python-dotenv
# MAGIC %restart_python

# COMMAND ----------

from importlib.metadata import version

print(f"langchain: {version('langchain')}")
print(f"langgraph: {version('langgraph')}")
print(f"langchain-community: {version('langchain-community')}")
print(f"mlflow: {version('mlflow')}")


# COMMAND ----------

from typing import Annotated, TypedDict, Sequence, Literal
from langchain_core.messages import BaseMessage
from langchain_core.documents.base import Document
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

allowed_routes: Sequence[str] = ["code", "general", "genie"]

class Router(BaseModel):
  """
  A router that will route the question to the right agent. Questions about finance should route to genie.
  """
  route: Literal[tuple(allowed_routes)] = Field(default="general", description=f"The route to follow should be one of: {', '.join(allowed_routes)}")


class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]
  context: Sequence[Document]
  route: str

# COMMAND ----------

from pprint import pprint

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex 

endpoint=""
index_name=""

vector_search = VectorSearchClient()

vector_search_index: VectorSearchIndex = vector_search.get_index(endpoint_name=endpoint, index_name=index_name)

result = vector_search_index.similarity_search(["source", "content"], query_text="We can use your drink machine how do we order it OBO?", num_results=1)

# results = vector_search.similarity_search(
#     query="What is OBO?", k=1, filter={}
# )
pprint(result)


# COMMAND ----------

from langchain.output_parsers.json import SimpleJsonOutputParser
import pandas as pd

space_id = ""

content = "show me total bank accounts for each country"
genie: Genie = Genie(space_id=space_id)
genie_response: GenieResponse = genie.ask_question(content)
# description: str = genie_response.description
result: str = genie_response.result
# response: str = f"{description}\n{result}"

llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

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

from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.vectorstores.base import VectorStore
from langchain_core.documents.base import Document

from databricks_langchain import ChatDatabricks
from databricks_langchain import DatabricksVectorSearch
from databricks_langchain.genie import Genie
from databricks_ai_bridge.genie import GenieResponse

import mlflow

@mlflow.trace()
def vector_search_question(state: AgentState) -> dict[str, str]:
    print(vector_search_question)
    messages: Sequence[BaseMessage] = state["messages"]
    content: str = messages[-1].content

    vector_search: VectorStore = DatabricksVectorSearch(
        endpoint=endpoint,
        index_name=index_name,
    )

    context: Sequence[Document] = vector_search.similarity_search(
        query=content, k=1, filter={}
    )

    return {"context": context}

@mlflow.trace()
def genie_question(state: AgentState) -> dict[str, str]:
    print(genie_question)
    messages: Sequence[BaseMessage] = state["messages"]
    content: str = messages[-1].content

    genie: Genie = Genie(space_id=space_id)
    genie_response: GenieResponse = genie.ask_question(content)
    description: str = genie_response.description
    result: str = genie_response.result
    response: str = f"{description}\n{result}"
    return {"messages": response}
  
@mlflow.trace()
def route_question(state: AgentState) -> dict[str, str]:
    print("route_question")
    llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
    chain: RunnableSequence = llm.with_structured_output(Router)
    messages: Sequence[BaseMessage] = state["messages"]
    last_message: BaseMessage = messages[-1]
    response = chain.invoke([last_message])
    return {"route": response.route}

# Creating the code agent that could be way more technical
@mlflow.trace()
def answer_code_question(state: AgentState) -> dict[str, BaseMessage]:
    llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
    prompt: PromptTemplate = PromptTemplate.from_template(
        "You are a software engineer. Answer this question with step by steps details : {input}"
    )
    chain: RunnableSequence = prompt | llm
    messages: Sequence[BaseMessage] = state["messages"]
    content: str = messages[-1].content
    response = chain.invoke({"input": content})
    return {"messages": response}

@mlflow.trace()
def answer_generic_question(state: AgentState) -> dict[str, BaseMessage]:
    llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
    prompt: PromptTemplate = PromptTemplate.from_template(
        "Give a general and concise answer to the question: {input}"
    )
    chain: RunnableSequence = prompt | llm
    messages: Sequence[BaseMessage] = state["messages"]
    content: str = messages[-1].content
    response = chain.invoke({"input": content})
    return {"messages": response}
  

@mlflow.trace()
def summarize_response(state: AgentState) -> dict[str, BaseMessage]:
    llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
    prompt: PromptTemplate = PromptTemplate.from_template(
        "Summarize: {input}"
    )
    chain: RunnableSequence = prompt | llm
    messages: Sequence[BaseMessage] = state["messages"]
    response = chain.invoke(messages)
    return {"messages": response}

# COMMAND ----------

from typing import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph


#Here is a simple 3 steps graph that is going to be working in the bellow "decision" condition
def create_graph() -> CompiledStateGraph:
    workflow: StateGraph = StateGraph(AgentState)

    workflow.add_node("router", route_question)
    workflow.add_node("code_agent", answer_code_question)
    workflow.add_node("generic_agent", answer_generic_question)
    workflow.add_node("genie_agent", genie_question)
    workflow.add_node("summarize_agent", summarize_response)

    workflow.add_conditional_edges(
        "router",
        lambda x: x["route"],
        {
            "code": "code_agent",
            "general": "generic_agent",
            "genie": "genie_agent"
        }
    )

    workflow.set_entry_point("router")
    workflow.add_edge("code_agent", "summarize_agent")
    workflow.add_edge("generic_agent", "summarize_agent")
    workflow.add_edge("genie_agent", "summarize_agent")
    workflow.set_finish_point("summarize_agent")
 
    
    return workflow.compile()

# COMMAND ----------

from langgraph.graph.state import CompiledStateGraph

graph: CompiledStateGraph = create_graph()

# COMMAND ----------

from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

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


