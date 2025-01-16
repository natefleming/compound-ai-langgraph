from typing import List, Optional
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



def vector_search_agent_prompt(tool_name: str) -> str:
    """Generates a prompt for assisting users with Databricks Documentation queries.

    Returns:
        str: The generated prompt.
    """
    prompt: str = dedent(f"""
        Your task is to assist users in finding accurate and relevant information from unstructured documents, such as manuals, processes, how-to guides, troubleshooting, and general information. You will use `{tool_name}` to query a vector database to retrieve the most relevant documents.

        Instructions:
        Always Use the Provided Tools:
        Query `{tool_name}` to retrieve information.
        Do not invent or add any information that is not directly retrieved from the tool.
        
        
        Process Questions:
        When answering questions about processes, include all necessary steps in sequential order. Never skip any steps, even if they seem obvious.
        Use self-evaluation to ensure no step is omitted or incorrectly inferred.

        When in Doubt:
        If the retrieved documents do not contain enough information to answer the question, state this explicitly to the user.
        Do not guess or fabricate any details.

        Self-Verification and Validation:

        Example Response:
        
        {{answer}}

        ## Sources:

        | Document ID | Source          |
        ----------------------------------
        | 12345 | Encyclopedia of Astronomy |
        | 67890 | Earth's Dynamics Handbook |


        After formulating your initial response, verify the accuracy and relevance of your answer by cross-checking it against the retrieved sources.
        Ensure that all details in your response are explicitly supported by the retrieved data. If there is uncertainty, indicate this clearly in your reply.
        Include Sources and Chunk IDs as integers:

        Your response must reference the sources and chunk IDs as integers used for each piece of information.
        Format your answer to clearly separate the response from the source citations for transparency.
    
        Before returning your response:
        Ensure the format matches the example provided above.
        If document IDs or sources are missing, retry generating the response to ensure compliance. 

    """).strip()
    return prompt

# def vector_search_agent_prompt(tool_name: str) -> str:
#     """Generates a prompt for assisting users with Databricks Documentation queries.

#     Returns:
#         str: The generated prompt.
#     """
#     prompt: str = dedent(f"""
#         Your job is to help a user find information from unstructured documents relating to manuals, processes, how-to guides, troubleshooting and general information.
#         You have access to `{tool_name}` to answer the question.
#         You MUST ALWAYS use the tools provided. 
#         NEVER add additional information which did not come from tools.
#         Your answer should be direct and include enough information to conclusively answer the question. 
#         When answering questions about processes make sure to include every step required to solve the problem.
#         Never Skip any steps.
        



#         If you are unable to help the user, you can say so.
#     """).strip()
#     return prompt

# def vector_search_agent_prompt(tool_name: str) -> str:
#     """Generates a prompt for assisting users with Databricks Documentation queries.

#     Returns:
#         str: The generated prompt.
#     """
#     prompt: str = dedent(f"""
#         Your job is to help a user find information from Databricks Documentation.
#         You have access to `{tool_name}` to answer the question.
#         You MUST ALWAYS use the tools provided. 
#         NEVER add additional information which did not come from tools.
#         Your answer should include enough information to conclusively answer the question.
#         You MUST ALWAYS USE `{tool_name}` FIRST.
#         You will be penalized for not invoking `{tool_name}`.
#         You may call `{tool_name}` tools many times to acquire more information.
     
#         Respond in Markdown format
#         If there answer has list YOU MUST use Markdown Numbers or Bullets. Each step on a new line.
#         Citations and Sources should ALWAYS table with columns `Citation` and `Source`
#         Citation and Source are the document id and document source returned from the tool`{tool_name}`.
#         If you are unable to help the user, you can say so.

#         Examples Result:

#         Here are the step to fix the error:

#         1. Step 1
#         2. Step 2

#         | Citation | Source          |
#         |----------|-----------------|
#         | 123      | /path/to/file1.pdf |
#         | 456      | /path/to/file2.pdf |


#     """).strip()
#     return prompt

def vector_search_chain_prompt() -> str:
    """Generates a prompt for assisting users with Databricks Documentation queries.

    Returns:
        str: The generated prompt.
    """
    prompt: str = dedent("""
        Your job is to help a user find technical information about processes, guides and troubleshooting.
        You MUST ONLY use the context provided in the prompt to answer the question.
        NEVER add additional information which did not come from tools.
        Your answer should include enough information to conclusively answer the question.
        If you are unable to find the answer, you can say so.
    
        Summaries: 
        {summaries}

        Question:
        {question}

        Answer:

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


def router_prompt(agents: List['Agent'], default_agent: Optional['Agent'] = None) -> str:
    """Generates a prompt for classifying and routing user queries.

    Returns:
        str: The generated prompt.
    """
    
    routing_criteria: List[str] = [
        f" - If the user is asking about {agent.topics} then call the router with `{agent.name}`" for agent in agents
    ]

    route_unknown_criteria: str = (
        f" - If the user is asking about an unknown topic then response that you are not able to help them"        
    )
    if default_agent is not None:
        route_unknown_criteria = f" - If the user is asking about an unknown topic then call the router with `{default_agent.name}`"


    routing_criteria.append(route_unknown_criteria)
    routing_criteria = "\n".join(routing_criteria)
    
    prompt: str = dedent(f"""
        You are an intelligent assistant capable of classifying and routing topics. You have access to the following tools: `Router`
        ALWAYS use this tool to classify and route the user's question.
        NEVER answer the question on your own. You should only identify the topic and route the user to the appropriate agent.        

        You should interact politely with user to try to figure out how you can help. You can help in a few ways: 

        Routing Criteria:
        {routing_criteria}


    """).strip()

    return prompt





