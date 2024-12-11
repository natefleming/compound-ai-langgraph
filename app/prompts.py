from typing import List
from textwrap import dedent

from langchain.tools import Tool


def unity_catalog_prompt() -> str:
    """Generates a prompt for assisting users with Databricks Warehouse, Unity Catalog, or Python-related queries.

    Returns:
        str: The generated prompt.
    """
    prompt: str = dedent("""
        Your job is to help a user find information from the Databricks Warehouse, Unity Catalog, UC or anything that can be solved with Python.
        You only have certain tools you can use.
        If you are unable to help the user, you can say so.
    """).strip()
    return prompt


def vector_search_prompt(tool_name: str) -> str:
    """Generates a prompt for assisting users with Databricks Documentation queries.

    Returns:
        str: The generated prompt.
    """
    prompt: str = dedent(f"""
        Your job is to help a user find information from Databricks Documentation.
        You have access to `{tool_name}` to answer the question.
        You MUST ALWAYS use the tools provided. 
        NEVER add additional information which did not come from tools.
        If you are unable to help the user, you can say so.
    """).strip()
    return prompt


def genie_prompt(tool_name: str) -> str:
    """Generates a prompt for assisting users with Genie-related queries.

    Args:
        tool_name (str): The name of the tool to be used for answering the question.

    Returns:
        str: The generated prompt.
    """
    prompt: str = dedent(f"""
        Your job is to help a user find information from Genie.
        You have access to `{tool_name}` to answer the question.
        ALWAYS use the tools provided.
        The Genie tool requires specific parameters. Input parameters should be passed in a dictionary.
        Summarize the question, excluding any reference to genie or genie spaces.
        If you are unable to help the user, you can say so.
    """).strip()
    return prompt


def router_prompt(agents: List['Agent']) -> str:
    """Generates a prompt for classifying and routing user queries.

    Returns:
        str: The generated prompt.
    """
    
    routing_criteria: List[str] = (
        f"- If the user is asking about {agent.description} then call the router with `{agent.name}`" for agent in agents
    )
    routing_criteria = "\n".join(routing_criteria)
    
    # prompt: str = dedent(f"""
    #     You are an intelligent assistant capable of classifying and routing topics.

    #     You should interact politely with user to try to figure out how you can help. You can help in a few ways:

    #     - If the user is asking about Databricks then call the router with `vector_search`
    #     - If the user is asking about genie then call the router with `genie`
    #     - If the user is asking about unity catalog, warehouse, python or match then call the router with `unity_catalog`
    #     - If you are unable to help the user, you can say so.
    # """).strip()
    prompt: str = dedent(f"""
        You are an intelligent assistant capable of classifying and routing topics.

        You should interact politely with user to try to figure out how you can help. You can help in a few ways:
        {routing_criteria}
        - If you are unable to help the user, you can say so.
    """).strip()
    return prompt