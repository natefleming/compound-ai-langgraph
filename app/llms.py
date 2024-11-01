from typing import Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_databricks import ChatDatabricks

EndpointOrAlias = str

def get_llm(
  model_name: EndpointOrAlias = "llama",
  base_url: Optional[str] = None,
  api_key: Optional[str] = None,
) -> BaseChatModel:
  match model_name:
    case "llama":
      llm: BaseChatModel = ChatDatabricks(
        endpoint="databricks-meta-llama-3-1-70b-instruct",
      )
    case "openai":
      llm: BaseChatModel = ChatOpenAI(
        model="gpt-4o",
      )
    case _:
      llm: BaseChatModel = ChatDatabricks(
        endpoint=model_name,
      )

  return llm