from typing import Callable, Dict, List, Optional, Union
from functools import partial

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.tools import Tool

from langgraph.graph import StateGraph, MessageGraph, END
from langgraph.graph.state import CompiledStateGraph

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from mlflow.langchain.output_parsers import (
    StringResponseOutputParser,
    ChatCompletionsOutputParser,
)

from app.router import route
from app.agents import Agent, create_agent
from app.prompts import router_prompt
from app.tools import create_router_tool
from app.messages import latest_message_content, first_message


def _display(self, verbose: bool = False) -> None:
  from IPython.display import Image, display
  display(Image(self.get_graph(xray=verbose).draw_mermaid_png()))

def _as_chain(self) -> RunnableSequence:
  chain: RunnableSequence = (
    #RunnableLambda(first_message) |
    RunnableLambda(partial(latest_message_content, self)) |
    ChatCompletionsOutputParser()
  )
  return chain

# monkey patch 
CompiledStateGraph.as_chain = _as_chain
CompiledStateGraph.display = _display

class GraphBuilder(object):

  def __init__(self, llm: BaseChatModel) -> None:
    self._llm = llm
    self._graph: StateGraph = MessageGraph()
    self._agents: List[Agent] = []
    self._memory: Optional[MemorySaver] = None
    self._debug: bool = False
    self._entry_point: str = "router"
    
  def add_agent(self, agent) -> 'GraphBuilder':
    self._agents.append(agent)
    return self

  def with_memory(self) -> 'GraphBuilder':
    self._memory = MemorySaver()
    return self
  
  def with_debug(self) -> 'GraphBuilder':
    self._debug = True
    return self
  
  def with_entry_point(self, node: Union[str|Agent]) -> 'GraphBuilder':
    if isinstance(node, Agent):
      self._entry_point = node.name
    else:
      self._entry_point = node
    return self
  
  def build(self) -> StateGraph:
    self.add_agent(self.router_agent())
    
    nodes: Dict[str, str] = {
      "tools": "tools",
      END: END,
    }
    
    for agent in self._agents:
      self._graph.add_node(agent.name, agent.as_runnable())
      nodes[agent.name] = agent.name

    for agent in self.agents:
      self._graph.add_conditional_edges(agent.name, route, nodes)

    self._graph.add_node("tools", ToolNode(self.tools))   
    self._graph.add_conditional_edges("tools", route, nodes)

    self._graph.set_entry_point(self._entry_point) 
        
    compiled_state_graph: CompiledStateGraph = (
      self._graph.compile(checkpointer=self._memory, debug=self._debug)
    )

    return compiled_state_graph


  def router_agent(self) -> Agent:
    prompt: str = router_prompt()
    router_tool: Tool = create_router_tool(choices=self.agents)

    router_agent: Agent = (
      create_agent(
        name="router",
        llm=self._llm, 
        prompt=prompt, 
        tools=[router_tool]
      )
    )

    return router_agent
  
  @property
  def agents(self) -> List[Agent]:
    return self._agents
  
  @property
  def tools(self) -> List[Tool]:
    agent_tools: List[Tool] = []
    for agent in self._agents:
      agent_tools.extend(agent.tools)
    return agent_tools
  

  