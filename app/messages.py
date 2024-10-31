from typing import Any, Dict, List

from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, BaseMessage

from app.state import AgentState

def first_message(payload: Dict[str, Any]) -> Dict[str, str]:
    print(f"first_message: {payload}")
    if "messages" not in payload:
        raise ValueError(f"No messages found in payload: {payload}")
    messages: List[Dict[str, Any]] = payload["messages"]
    return messages[0]

def latest_message_content(chain: Runnable, state: AgentState):
    messages: List[BaseMessage] = state["messages"]
    return chain.invoke(messages)[-1].content

def get_name(state: AgentState) -> str:
    return state["sender"]

def add_name(state: AgentState, name: str) -> AIMessage:
    state["sender"] = name
    return state

def remove_name(state: AgentState) -> AIMessage:
    state.pop("name", None)
    return state