from typing import Any, Dict, List

from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, BaseMessage


def first_message(payload: Dict[str, Any]) -> Dict[str, str]:
    if "messages" not in payload:
        raise ValueError(f"No messages found in payload: {payload}")
    messages: List[Dict[str, Any]] = payload["messages"]
    return messages[0]

def latest_message_content(chain: Runnable, messages: List[BaseMessage]):
    return chain.invoke(messages)[-1].content

def get_name(message: AIMessage) -> str:
    attributes: Dict[str, Any] = message.dict()
    return attributes.get("name")

def add_name(message: AIMessage, name: str) -> AIMessage:
    attributes: Dict[str, Any] = message.dict()
    attributes["name"] = name
    return AIMessage(**attributes)


def remove_name(message: AIMessage) -> AIMessage:
    attributes: Dict[str, Any] = message.dict()
    attributes.pop("name", None)
    return AIMessage(**attributes)