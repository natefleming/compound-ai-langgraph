
from typing import Any, Dict, Optional, List, Callable, Union
from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.tools import Tool
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.messages import SystemMessage, BaseMessage

from app.router import filter_out_routes
from app.messages import add_name
from app.tools import create_genie_tool, create_vector_search_tool
from app.prompts import genie_prompt, vector_search_prompt


AgentOrString = Union['AgentBase', str]
Condition = Callable[[List[BaseMessage]], str]
RouteMapping = Dict[str, AgentOrString]
AgentOrConditionalRoute = Union['AgentBase', 'ConditionalRoute']


@dataclass
class ConditionalRoute:
  condition: Condition
  route_mapping: RouteMapping


class AgentBase(ABC):

  def __init__(self) -> None:
    self._direct_routes: List['AgentBase'] = []
    self._conditional_routes: List[ConditionalRoute] = []

  @abstractmethod
  def as_runnable() -> RunnableSequence:
    ...

  def then(
    self, 
    next_agent: AgentOrConditionalRoute
  ) -> 'AgentBase':
    if isinstance(next_agent, AgentBase):
      self._direct_routes.append(next_agent)
    else:
      self._conditional_routes.append(next_agent)
    return next

  @property
  def direct_routes(self) -> List['Agent']:
    return self._direct_routes

  @property
  def conditional_routes(self) -> List[ConditionalRoute]:
    return self._conditional_routes
  

class Agent(AgentBase):

  def __init__(
    self, 
    name: str, 
    llm: BaseChatModel, 
    prompt: Optional[str] = None, 
    tools: List[Tool] = []
  ) -> None:
    super().__init__()
    self.name = name
    self.llm = llm
    self.prompt = prompt
    self.tools = tools

  def as_runnable(self) -> RunnableSequence:
    def _get_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
      if self.prompt is not None:
        return [SystemMessage(content=self.prompt)] + messages
      else:
        return messages
    
    chain: RunnableSequence = (
      RunnableLambda(filter_out_routes) | 
      RunnableLambda(_get_messages) | 
      self.llm.bind_tools(self.tools) | 
      partial(add_name, name=self.name)
    )
    
    return chain

def create_agent(
    name: str, 
    llm: BaseChatModel, 
    prompt: Optional[str] = None, 
    tools: List[Tool] = []
) -> Agent:
  return Agent(name=name, llm=llm, prompt=prompt, tools=tools)



def create_genie_agent(
    llm: BaseChatModel,
    space_id: str, 
    workspace_host: Optional[str] = None,
    token: Optional[str] = None,
) -> Agent:
    
    genie_tool: Tool = create_genie_tool(
        space_id=space_id, 
        workspace_host=workspace_host,
        token=token
    )
    prompt: str = genie_prompt(genie_tool.name)

    genie_agent: Agent = create_agent(
        name="genie", 
        llm=llm, 
        prompt=prompt, 
        tools=[genie_tool]
    )

    return genie_agent


def create_vector_search_agent(
  llm: BaseChatModel,
  endpoint_name: str,
  index_name: str,
  columns: List[str] = None,
  parameters: Dict[str, Any] = None
) -> Agent:

  vector_search_tool: Tool = (
    create_vector_search_tool(
      endpoint_name=endpoint_name,
      index_name=index_name,
      columns=columns,
      parameters=parameters,
    )
  )

  prompt: str = vector_search_prompt()

  vector_search_agent: Agent = (
    create_agent(
      name="vector_search",
      llm=llm, 
      prompt=prompt, 
      tools=[vector_search_tool]
    )
  )

  return vector_search_agent
