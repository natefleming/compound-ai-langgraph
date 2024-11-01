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
from app.prompts import router_prompt
from app.tools import create_router_tool
from app.messages import latest_message_content, first_message
from app.agents import (
    AgentBase, 
    ConditionalRoute, 
    AgentOrString,
    create_agent
)


def _display(self, verbose: bool = False) -> None:
    from IPython.display import Image, display

    display(Image(self.get_graph(xray=verbose).draw_mermaid_png()))


def _as_chain(self, *, first_message_only: bool = False) -> RunnableSequence:
    """
    Constructs a runnable sequence (chain) based on the specified condition.

    Args:
        first_message_only (bool, optional): 
            If True, the function will create a chain that only processes the first message.
            If False, the function will process all messages using the instance's default behavior.
            Defaults to False.

    Returns:
        RunnableSequence: 
            A chain of processing steps for either the first message only or for all messages.
    """
    chain: RunnableSequence = (
        RunnableLambda(first_message) if first_message_only else self
    )
    chain = (
        RunnableLambda(partial(latest_message_content, self))
        | ChatCompletionsOutputParser()
    )
    return chain


# monkey patch
CompiledStateGraph.as_chain = _as_chain
CompiledStateGraph.display = _display


class GraphBuilder(object):
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._graph: StateGraph = MessageGraph()
        self._agents: List[AgentBase] = []
        self._memory: Optional[MemorySaver] = None
        self._debug: bool = False
        self._entry_point: str = "router"

    def add_agent(self, agent: AgentBase) -> "GraphBuilder":
        self._agents.append(agent)
        return self

    def with_memory(self) -> "GraphBuilder":
        self._memory = MemorySaver()
        return self

    def with_debug(self) -> "GraphBuilder":
        self._debug = True
        return self

    def with_entry_point(self, node: AgentOrString) -> "GraphBuilder":
        if isinstance(node, AgentBase):
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
            for conditional_route in agent.conditional_routes:
                conditional_route: ConditionalRoute
                self._graph.add_conditional_edges(
                    agent.name, 
                    conditional_route.condition, 
                    conditional_route.route_mapping
                )
            for direct_route in agent.direct_routes:
                direct_route: AgentBase
                self._graph.add_node(direct_route.as_runnable())
                self._graph.add_edge(agent.name, direct_route.name)

        self._graph.add_node("tools", ToolNode(self.tools))
        self._graph.add_conditional_edges("tools", route, nodes)

        self._graph.set_entry_point(self._entry_point)

        compiled_state_graph: CompiledStateGraph = self._graph.compile(
            checkpointer=self._memory, debug=self._debug
        )

        return compiled_state_graph

    def router_agent(self) -> AgentBase:
        prompt: str = router_prompt()
        router_tool: Tool = create_router_tool(choices=self.agents)

        router_agent: AgentBase = create_agent(
            name="router", llm=self._llm, prompt=prompt, tools=[router_tool]
        )

        return router_agent

    @property
    def agents(self) -> List[AgentBase]:
        return self._agents

    @property
    def tools(self) -> List[Tool]:
        agent_tools: List[Tool] = []
        for agent in self._agents:
            agent_tools.extend(agent.tools)
        return agent_tools