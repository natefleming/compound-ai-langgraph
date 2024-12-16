from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.tools import Tool, tool
from langchain_core.tools import StructuredTool
from langchain_databricks.vectorstores import DatabricksVectorSearch
from langchain_community.tools.databricks import UCFunctionToolkit

from databricks_langchain.genie import Genie

from app.retrievers import create_vector_search_retriever


# def format_cited_answers(
#     answer: str,
#     sources: List[str],
#     citations: List[int],
# ) -> str:
#     """
#     Formats the answer, source IDs, and sources in markdown and returns the string.
#     The IDs and source locations are formatted in a markdown table.
#     """
#     from textwrap import dedent

#     # Start with the answer section
#     md_output = f"## Answer {answer}\n\n"

#     # Add the sources table
#     md_output += "| Citation | Source |\n"
#     md_output += "|----|--------|\n"

#     for source, citation in zip(sources, citations):
#         md_output += f"| {citation} | {source} |\n"

#     return md_output

def create_cited_answer_tool() -> Tool:
    
    class CitedAnswer(BaseModel):
        """
            Answer the user question based only on the given sources, and cite the sources used.
            This tool is intended to be used to format results from Documents
        """

        answer: str = Field(
            ...,
            description="The answer to the user question, which is based only on the given sources.",
        )
        sources: List[str] = Field(
            ...,
            description="The integer IDs of the SPECIFIC sources which justify the answer.",
        )
        citations: List[int] = Field(
            default=list,
            description="The numeric IDs of the SPECIFIC sources which justify the answer.",
        )

    # tool = StructuredTool.from_function(
    #     func=format_cited_answers,
    #     name="format_cited_answer",
    #     description="formats a cited answer",
    #     args_schema=CitedAnswer,
    #     return_direct=True,
    #     # coroutine= ... <- you can specify an async method if desired as well
    # )
    return CitedAnswer
    

def create_unity_catalog_tools(warehouse_id: str, functions: List[str]) -> List[Tool]:
    """Creates a list of Unity Catalog tools.

    Args:
        warehouse_id (str): The ID of the warehouse.
        functions (List[str]): A list of function names to include.

    Returns:
        List[Tool]: A list of tools created from the Unity Catalog functions.
    """
    uc_function_toolkit: UCFunctionToolkit = UCFunctionToolkit(
        warehouse_id=warehouse_id
    ).include(*functions)
    return uc_function_toolkit.get_tools()


def create_vector_search_tool(
    endpoint_name: str,
    index_name: str,
    name: str = "vector_search_tool",
    description: str = "This tool is used to conduct similarity search using Databricks Vector Search",
    primary_key: Optional[str] = None,
    text_column: Optional[str] = None,
    doc_uri: Optional[str] = None,
    columns: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Tool:
    """Creates a vector search tool.

    Args:
        endpoint_name (str): The name of the endpoint.
        index_name (str): The name of the index.
        name (str, optional): The name of the tool. Defaults to "vector_search_tool".
        description (str, optional): The description of the tool. Defaults to "This tool is used to conduct similarity search using Databricks Vector Search".
        text_column (Optional[str], optional): The text column to use. Defaults to None.
        columns (Optional[List[str]], optional): The columns to use. Defaults to None.
        parameters (Optional[Dict[str, Any]], optional): Additional parameters. Defaults to None.

    Returns:
        Tool: The created vector search tool.
    """

    vector_search_as_retriever: VectorStoreRetriever = create_vector_search_retriever(
        endpoint_name=endpoint_name,
        index_name=index_name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        columns=columns,
        parameters=parameters,
    )
    
    vector_search_tool: Tool = vector_search_as_retriever.as_tool(
        name=name, description=description
    )

    return vector_search_tool


def create_genie_tool(
    space_id: Optional[str] = None,
    workspace_host: Optional[str] = None,
    token: Optional[str] = None,
) -> Tool:
    """Creates a Genie tool.

    Args:
        space_id (Optional[str], optional): The space ID. Defaults to None.
        workspace_host (Optional[str], optional): The workspace host. Defaults to None.
        token (Optional[str], optional): The token. Defaults to None.

    Returns:
        Tool: The created Genie tool.
    """
    space_id = space_id or os.environ.get("DATABRICKS_GENIE_SPACE_ID")
    genie_client: Genie = Genie(
        space_id=space_id,
    )

    @tool("genie_tool")
    def genie_tool(question: str) -> str:
        """
        This tool lets you have a conversation and chat with tabular data about <topic>. You should ask
        questions about the data and the tool will try to answer them.
        Please ask simple clear questions that can be answer by sql queries. If you need to do statistics or other forms of testing defer to using another tool.
        Try to ask for aggregations on the data and ask very simple questions. Prefer to call this tool multiple times rather than asking a complex question.

        Args:
            question: The question to ask to ask Genie

        Returns:
            str: An object containing the Genie response
        """
        answer: str = genie_client.ask_question(question)
        return answer

    return genie_tool


def create_router_tool(choices: List["Agent"]) -> Tool:
    """Creates a router tool.

    Args:
        choices (List['Agent']): A list of agent choices.

    Returns:
        Tool: The created router tool.
    """
    agent_names: List[str] = [a.name for a in choices]

    class Router(BaseModel):
        """
        Call this if you are able to route the user to the appropriate representative.
        """

        choice: str = Field(description=f"should be one of: {','.join(agent_names)}")

    return Router
