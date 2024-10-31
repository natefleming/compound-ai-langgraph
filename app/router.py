
from typing import List

from langchain_core.messages import AIMessage, BaseMessage

from langgraph.graph import END


def filter_out_routes(messages: List[BaseMessage]) -> List[BaseMessage]:
    filtered_messages: List[BaseMessage] = []
    for m in messages:
        if is_tool_call(m):
            if m.name == "router":
                continue
        filtered_messages.append(m)
    return filtered_messages
  
  
def get_last_ai_message(messages: List[BaseMessage]) -> AIMessage:
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None


def is_tool_call(message: BaseMessage) -> bool:
    return hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs


def route(messages: List[BaseMessage]) -> str:
    last_message = messages[-1]
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
    # if last_message.name == "vector_search":
    #     return "vector_search"
    else:
        return "router"
      

      