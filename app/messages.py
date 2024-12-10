from typing import Any, Dict, List

from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, BaseMessage

from guardrails.guard import Guard
from guardrails.classes.validation_outcome import ValidationOutcome


def apply_guard(messages: List[AIMessage], guard: Guard) -> AIMessage:
    last_message: AIMessage = messages[-1]
    validation_outcome: ValidationOutcome = guard.validate(last_message.content)
    if not validation_outcome.validation_passed:
        failure_reasons: List[str] = (
            [summary.failure_reason for summary in validation_outcome.validation_summaries]
        )
        failure_reasons = "; ".join(failure_reasons)
        guard_message: AIMessage = AIMessage(content=f"Guard failed - {failure_reasons}")
        # Forward sender to the guard message
        if hasattr(last_message, "name"):
            add_name(guard_message, last_message.name)
        messages.append(guard_message)

    return messages
    
        
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