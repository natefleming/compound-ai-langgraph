from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from databricks_langchain import ChatDatabricks

EndpointOrAlias = str

def get_llm(
    model_name: EndpointOrAlias = "llama",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Get a language model instance based on the model name.

    Args:
        model_name (EndpointOrAlias): The name or endpoint of the model. Defaults to "llama".
        base_url (Optional[str]): The base URL for the API. Defaults to None.
        api_key (Optional[str]): The API key for authentication. Defaults to None.

    Returns:
        BaseChatModel: An instance of a language model.
    """
    match model_name:
        case "llama":
            llm: BaseChatModel = ChatDatabricks(
                endpoint="databricks-meta-llama-3-1-70b-instruct",
                temperature=0.1,
            )
        case "openai":
            llm: BaseChatModel = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
            )
        case _:
            llm: BaseChatModel = ChatDatabricks(
                endpoint=model_name,
                temperature=0.1,
            )

    return llm