
from typing import List, Any, Dict

from langchain_core.messages import AIMessage, BaseMessage

from langgraph.graph import END

from app.state import AgentState

def filter_out_routes(state: AgentState) -> Dict[str, Any]:
    print(f"filter_out_routes: state={state}")
    messages: List[BaseMessage] = state["messages"]
    print(f"filter_out_routes: messages={messages}")
    sender: str = state.get("sender")
    print(f"filter_out_routes: sender={sender}")
    filtered_messages: List[BaseMessage] = []
    for m in messages:
        if is_tool_call(m):
            if sender == "router":
                continue
        filtered_messages.append(m)
    return {
        "messages": filtered_messages
    }
  
def get_last_ai_message(messages: List[BaseMessage]) -> AIMessage:
    print(f"get_last_ai_message: messages={messages}")
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None


def is_tool_call(message: BaseMessage) -> bool:
    return hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs


def route(state: AgentState) -> str:
    print(f"route: state={state}")
    messages: List[BaseMessage] = state["messages"]
    sender: str = state["sender"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if not last_message.tool_calls:
            return END
        else:
            if sender == "router":
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
      

      