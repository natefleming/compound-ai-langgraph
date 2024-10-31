import asyncio
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Annotated, Callable, List, Any, Mapping

import httpx
import pandas as pd
from langchain_core.tools import tool, InjectedToolArg, BaseTool, BaseToolkit
from pydantic import BaseModel, Field


@dataclass
class GenieResult:
    space_id: str
    conversation_id: str
    question: str
    content: Optional[str]
    sql_query: Optional[str] = None
    sql_query_description: Optional[str] = None
    sql_query_result: Optional[pd.DataFrame] = None
    error: Optional[str] = None


class GenieClient:

    def __init__(self, *,
                 host: Optional[str] = None,
                 token: Optional[str] = None,
                 api_prefix: str = "/api/2.0/genie/spaces"):
        self.host = host or os.environ.get("DATABRICKS_HOST")
        self.token = token or os.environ.get("DATABRICKS_TOKEN")
        assert self.host is not None, "DATABRICKS_HOST is not set"
        assert self.token is not None, "DATABRICKS_TOKEN is not set"
        self._workspace_client = httpx.AsyncClient(base_url=self.host,
                                                   headers={"Authorization": f"Bearer {self.token}"},
                                                   timeout=30.0)
        self.api_prefix = api_prefix
        self.max_retries = 300
        self.retry_delay = 1
        self.new_line = "\r\n"

    async def start(self, space_id: str, start_suffix: str = "") -> str:
        path = f"{self.api_prefix}/{space_id}/start-conversation"
        resp = await self._workspace_client.post(
            url=path,
            headers={"Content-Type": "application/json"},
            json={"content": "starting conversation" if not start_suffix else f"starting conversation {start_suffix}"},
        )
        resp = resp.json()
        return resp["conversation_id"]

    async def ask(self, space_id: str, conversation_id: str, message: str) -> GenieResult:
        path = f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages"
        # TODO: cleanup into a separate state machine
        resp_raw = await self._workspace_client.post(
            url=path,
            headers={"Content-Type": "application/json"},
            json={"content": message},
        )
        resp = resp_raw.json()
        message_id = resp.get("message_id", resp.get("id"))
        if message_id is None:
            print(resp, resp_raw.url, resp_raw.status_code, resp_raw.headers)
            return GenieResult(content=None, error="Failed to get message_id")

        attempt = 0
        query = None
        query_description = None
        content = None

        while attempt < self.max_retries:
            resp_raw = await self._workspace_client.get(
                f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages/{message_id}",
                headers={"Content-Type": "application/json"},
            )
            resp = resp_raw.json()
            status = resp["status"]
            if status == "COMPLETED":
                try:

                    query = resp["attachments"][0]["query"]["query"]
                    query_description = resp["attachments"][0]["query"].get("description", None)
                    content = resp["attachments"][0].get("text", {}).get("content", None)
                except Exception as e:
                    return GenieResult(
                        space_id=space_id,
                        conversation_id=conversation_id,
                        question=message,
                        content=resp["attachments"][0].get("text", {}).get("content", None)
                    )
                break

            elif status == "EXECUTING_QUERY":
                await self._workspace_client.get(
                    f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages/{message_id}/query-result",
                    headers={"Content-Type": "application/json"},
                )
            elif status in ["FAILED", "CANCELED"]:
                return GenieResult(
                    space_id=space_id,
                    conversation_id=conversation_id,
                    question=message,
                    content=None,
                    error=f"Query failed with status {status}"
                )
            elif status != "COMPLETED" and attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
            else:
                return GenieResult(
                    space_id=space_id,
                    conversation_id=conversation_id,
                    question=message,
                    content=None,
                    error=f"Query failed or still running after {self.max_retries * self.retry_delay} seconds"
                )
            attempt += 1
        resp = await self._workspace_client.get(
            f"{self.api_prefix}/{space_id}/conversations/{conversation_id}/messages/{message_id}/query-result",
            headers={"Content-Type": "application/json"},
        )
        resp = resp.json()
        columns = resp["statement_response"]["manifest"]["schema"]["columns"]
        header = [str(col["name"]) for col in columns]
        rows = []
        output = resp["statement_response"]["result"]
        if not output:
            return GenieResult(
                space_id=space_id,
                conversation_id=conversation_id,
                question=message,
                content=content,
                sql_query=query,
                sql_query_description=query_description,
                sql_query_result=pd.DataFrame([], columns=header),
            )
        for item in resp["statement_response"]["result"]["data_typed_array"]:
            row = []
            for column, value in zip(columns, item["values"]):
                type_name = column["type_name"]
                str_value = value.get("str", None)
                if str_value is None:
                    row.append(None)
                    continue
                match type_name:
                    case "INT" | "LONG" | "SHORT" | "BYTE":
                        row.append(int(str_value))
                    case "FLOAT" | "DOUBLE" | "DECIMAL":
                        row.append(float(str_value))
                    case "BOOLEAN":
                        row.append(str_value.lower() == "true")
                    case "DATE":
                        row.append(datetime.strptime(str_value, "%Y-%m-%d").date())
                    case "TIMESTAMP":
                        row.append(datetime.strptime(str_value, "%Y-%m-%d %H:%M:%S"))
                    case "BINARY":
                        row.append(bytes(str_value, "utf-8"))
                    case _:
                        row.append(str_value)
            rows.append(row)

        query_result = pd.DataFrame(rows, columns=header)
        return GenieResult(
            space_id=space_id,
            conversation_id=conversation_id,
            question=message,
            content=content,
            sql_query=query,
            sql_query_description=query_description,
            sql_query_result=query_result,
        )


class GenieInput(BaseModel):
    conversation_mapping: Annotated[Mapping[str, str], InjectedToolArg] = Field(default_factory=dict)
    question: str = Field(description="question to ask the genie data room. only ask relevant question to the appropriate room. do not ask for data that probably wont be in the room. "
                                   "make sure this is detailed and contains both the question and summarized history (omit if there is no chat history) "
                                   "in the format:\nHistory: <history>\nQuestion: <question>")


def lc_genie_space(genie_space_func: Optional[Callable] = None,
                   space_id: Optional[str] = None,
                   genie_client: Optional[GenieClient] = None):
    space_id = space_id or os.environ.get("DATABRICKS_GENIE_SPACE_ID")
    assert space_id is not None, ("DATABRICKS_GENIE_SPACE_ID is not set nor passed as an "
                                  "argument to decorator: lc_genie_space")

    def outer(func: Callable):
        async def inner(question: str, conversation_mapping: Annotated[Mapping[str, str], InjectedToolArg] = {}):
            client = genie_client or GenieClient()
            # check if conversation exists
            conversation_id = conversation_mapping.get(space_id)
            if conversation_id is None:
                conversation_id = await client.start(space_id, start_suffix=f"{str(datetime.now())}-{str(uuid.uuid4())}")

            result = await client.ask(space_id, conversation_id, question)
            return await func(question, result)

        inner.__doc__ = func.__doc__
        inner.__name__ = func.__name__
        return tool(inner, args_schema=GenieInput)

    if genie_space_func is not None:
        if callable(genie_space_func):
            return outer(genie_space_func)

    return outer

class GenieSpaceConfig(BaseModel):
    space_id: str
    space_name: str
    space_description: str
    genie_client: Optional[GenieClient] = None

    class Config:
        arbitrary_types_allowed = True

    def to_tool(self):
        def _tmp(query: str, result: GenieResult):
            return result
        _tmp.__name__ = self.space_name
        _tmp.__doc__ = self.space_description
        return lc_genie_space(
            genie_space_func=_tmp,
            space_id=self.space_id,
            genie_client=self.genie_client
        )

def is_valid_name(s):
    """
    Check if the string `s` does not start with a number and only contains alphanumeric characters or underscores.

    :param s: The string to check.
    :return: True if the string is valid, False otherwise.
    """
    # Define the regular expression pattern
    pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$'

    # Match the pattern
    return bool(re.match(pattern, s))

class GenieToolkit(BaseToolkit):

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        self._space_configs: List[GenieSpaceConfig] = []

    def with_space_id(self,
                      *,
                      space_id: str,
                      space_name: str,
                      space_description: str,
                      genie_client: Optional[GenieClient] = None):
        if not is_valid_name(space_name):
            raise ValueError("The space name must start with a letter and "
                             "contain only alphanumeric characters or underscores")
        self._space_configs.append(
            GenieSpaceConfig(
                space_id=space_id,
                space_name=space_name,
                space_description=space_description,
                genie_client=genie_client
            )
        )
        return self

    def get_tools(self) -> List[BaseTool]:
        if len(self._space_configs) == 0:
            raise ValueError("No tools have been added to the toolkit")
        return [space.to_tool() for space in self._space_configs] # noqa