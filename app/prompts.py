from typing import List
from textwrap import dedent

from langchain.tools import Tool


def vector_search_prompt() -> str:
  prompt: str = dedent("""
    Your job is to help a user find information from Databricks Documentation.
    You only have certain tools you can use.
    If you are unable to help the user, you can say so.
    """).strip()
  return prompt


def genie_prompt(tool_name: str) ->  str: 
  prompt: str = dedent(f"""
    Your job is to help a user find information from Genie.
    You have access to `{tool_name}` to answer the question.
    ALWAYS use the tools provided. 
    The Genie tool requires specific parameters. Input parameters should be passed in a dictionary.
    Summarize the question, excluding any reference to genie or genie spaces
    If you are unable to help the user, you can say so.
    """).strip()
  return prompt


def router_prompt() -> str:
  prompt: str = dedent("""
    You are an intelligent assistant capable of classifying and routing topics.

    You should interact politely with user to try to figure out how you can help. You can help in a few ways:

    - If the user is asking about Databricks then call the router with `vector_search`
    - If the user is asking about genie then call the router with `genie`
    - If you are unable to help the user, you can say so.
    """).strip()
  return prompt


