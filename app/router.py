from typing import List
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END

from app.messages import get_last_ai_message

def filter_out_routes(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Filters out messages that are tool calls to the router.

    Args:
        messages (List[BaseMessage]): List of messages to filter.

    Returns:
        List[BaseMessage]: Filtered list of messages.
    """ 
    filtered_messages: List[BaseMessage] = []
    for m in messages:
        if is_tool_call(m):
            if m.name == "router":
                continue
        filtered_messages.append(m)
    return filtered_messages

def has_function_call(message: BaseMessage, function_name: str) -> bool:
    if not is_tool_call(message):
        return False
    tool_calls = message.additional_kwargs["tool_calls"]
    for tool_call in tool_calls:
        if "function" in tool_call:
            name: str = tool_call["function"]["name"]
            if name == function_name:
                return True
    return False


def is_tool_call(message: BaseMessage) -> bool:
    """Checks if a message is a tool call.

    Args:
        message (BaseMessage): The message to check.

    Returns:
        bool: True if the message is a tool call, False otherwise.
    """
    return hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs


def route(messages: List[BaseMessage], agent_names: List[str]) -> str:
    """Determines the route based on the last message in the list.

    Args:
        messages (List[BaseMessage]): List of messages to route.

    Returns:
        str: The determined route.
    """
    last_message: BaseMessage = messages[-1]
    if isinstance(last_message, AIMessage):
        if not last_message.tool_calls:
            return END
        else:
            if last_message.name == "router":
                if len(last_message.tool_calls) > 1:
                    raise ValueError("Too many tools")
                return last_message.tool_calls[0]["args"]["choice"]
            else:
                return "tools"
    last_message: AIMessage = get_last_ai_message(messages)
    if last_message is None:
        return "router"
    
    if last_message.name == "vector_search" and has_function_call(last_message, "CitedAnswer"):
        return END

        
    if last_message.name in agent_names:
        return last_message.name
    # if last_message.name == "vector_search":
    #     return "vector_search"
    # if last_message.name == "genie":
    #     return "genie"
    # if last_message.name == "unity_catalog":
    #     return "unity_catalog"

    return END
    #return "router"