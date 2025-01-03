from typing import Any, Dict, List
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
#from guardrails.guard import Guard
#from guardrails.classes.validation_outcome import ValidationOutcome


def get_last_ai_message(messages: List[BaseMessage]) -> AIMessage:
    """Gets the last AIMessage from a list of messages.

    Args:
        messages (List[BaseMessage]): List of messages to search.

    Returns:
        AIMessage: The last AIMessage found, or None if not found.
    """
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None
    
def get_last_human_message(messages: List[BaseMessage]) -> HumanMessage:
    """Gets the last HumanMessage from a list of messages.

    Args:
        messages (List[BaseMessage]): List of messages to search.

    Returns:
        HumanMessage: The last HumanMessage found, or None if not found.
    """
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            return m
    return None

# def apply_guard(messages: List[BaseMessage], guard: Guard) -> List[BaseMessage]:
#     """Applies guard validation to the last message in the list.

#     Args:
#         messages (List[BaseMessage]): List of base messages.
#         guard (Guard): Guard object for validation.

#     Returns:
#         List[BaseMessage]: List of base messages with guard validation applied to the last message.
#     """
#     last_message: AIMessage = messages[-1]
#     validation_outcome: ValidationOutcome = guard.validate(last_message.content)
#     if not validation_outcome.validation_passed:
#         failure_reasons: List[str] = (
#             [summary.failure_reason for summary in validation_outcome.validation_summaries]
#         )
#         failure_reasons = "; ".join(failure_reasons)
#         guard_message: AIMessage = AIMessage(content=f"Guard failed - {failure_reasons}")
#         if hasattr(last_message, "name"):
#             add_name(guard_message, last_message.name)
#         messages.append(guard_message)
#     return messages

def first_message(payload: Dict[str, Any]) -> Dict[str, str]:
    """Retrieves the first message from the payload.

    Args:
        payload (Dict[str, Any]): Payload containing messages.

    Returns:
        Dict[str, str]: The first message in the payload.

    Raises:
        ValueError: If no messages are found in the payload.
    """
    if "messages" not in payload:
        raise ValueError(f"No messages found in payload: {payload}")
    messages: List[Dict[str, Any]] = payload["messages"]
    return messages[0]

def latest_message_content(chain: Runnable, messages: List[BaseMessage]):
    """Gets the content of the latest message after invoking the chain.

    Args:
        chain (Runnable): Runnable chain object.
        messages (List[BaseMessage]): List of base messages.

    Returns:
        str: Content of the latest message.
    """
    return chain.invoke(messages)[-1].content

def get_name(message: AIMessage) -> str:
    """Retrieves the name attribute from the message.

    Args:
        message (AIMessage): AI message object.

    Returns:
        str: Name attribute of the message.
    """
    attributes: Dict[str, Any] = message.dict()
    return attributes.get("name")

def add_name(message: AIMessage, name: str) -> AIMessage:
    """Adds a name attribute to the message.

    Args:
        message (AIMessage): AI message object.
        name (str): Name to be added.

    Returns:
        AIMessage: Updated AI message with the name attribute.
    """
    attributes: Dict[str, Any] = message.dict()
    attributes["name"] = name
    return AIMessage(**attributes)

def remove_name(message: AIMessage) -> AIMessage:
    """Removes the name attribute from the message.

    Args:
        message (AIMessage): AI message object.

    Returns:
        AIMessage: Updated AI message without the name attribute.
    """
    attributes: Dict[str, Any] = message.dict()
    attributes.pop("name", None)
    return AIMessage(**attributes)