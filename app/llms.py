from typing import Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_databricks import ChatDatabricks

def get_llm(
  model_name: Literal["llama", "gpt4"] = "llama",
  base_url: Optional[str] = None,
  api_key: Optional[str] = None,
) -> BaseChatModel:
  match model_name:
    case "llama":
      llm: BaseChatModel = ChatDatabricks(
        endpoint="databricks-meta-llama-3-1-70b-instruct",
      )
    case "gpt4":
      llm: BaseChatModel = ChatOpenAI(
        model="gpt-4o",
      )
    case _:
      raise ValueError(f"Invalid model_name: {model_name}")

  return llm