
from typing import List, Dict, Any, Optional

from datetime import datetime
import uuid
import asyncio

from pydantic import BaseModel, Field

from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.tools import Tool, tool
from langchain_databricks.vectorstores import DatabricksVectorSearch

from app.genie_utils import GenieResult, lc_genie_space, GenieClient, GenieToolkit

# import nest_asyncio
# nest_asyncio.apply()


def create_vector_search_tool(
  endpoint_name: str,
  index_name: str,
  name: str = "vector_search_tool",
  description: str = "This tool is used to conduct similarity search using Databricks Vector Search",
  text_column: Optional[str] = None,
  columns: Optional[List[str]] = None,
  parameters: Optional[Dict[str, Any]] = None,
) -> Tool:
    
    vector_search: DatabricksVectorSearch = DatabricksVectorSearch(
        index_name=index_name,
        endpoint=endpoint_name,
        text_column=text_column,
        columns=columns,
    )
    vector_search_as_retriever: VectorStoreRetriever = (
        vector_search.as_retriever(search_kwargs=parameters)
    )
    vector_search_tool: Tool = vector_search_as_retriever.as_tool(
        name=name, 
        description=description
    )

    return vector_search_tool


def create_genie_tool(
    space_id: Optional[str] = None,
    workspace_host: Optional[str] = None,
    token: Optional[str] = None
) -> Tool:

    space_id = space_id or os.environ.get("DATABRICKS_GENIE_SPACE_ID")
    
    genie_client: GenieClient = GenieClient(
        host=workspace_host,  
        token=token, 
    )

    @tool("genie_tool")
    def genie_tool(question: str) -> GenieResult:
        """
        This tool lets you have a conversation and chat with tabular data about <topic>. You should ask
        questions about the data and the tool will try to answer them. 
        Please ask simple clear questions that can be answer by sql queries. If you need to do statistics or other forms of testing defer to using another tool.
        Try to ask for aggregations on the data and ask very simple questions. Prefer to call this tool multiple times rather than asking a complex question.

        Args:
            question: The question to ask to ask Genie

        Returns:
            GenieResult: An object containing the Genie response
        """
        import nest_asyncio
        nest_asyncio.apply()
        
        conversation_id: str = (
            asyncio.run(genie_client.start(
                space_id, 
                start_suffix=f"{str(datetime.now())}-{str(uuid.uuid4())}"
            ))
        )
        result: GenieResult = asyncio.run(genie_client.ask(space_id, conversation_id, question))
        return result
    
    return genie_tool


def create_router_tool(choices: List['Agent']) -> Tool:

    agent_names: List[str] = [a.name for a in choices]
    
    class Router(BaseModel):
        """
        Call this if you are able to route the user to the appropriate representative.
        """

        choice: str = Field(description=f"should be one of: {','.join(agent_names)}")

    return Router
  