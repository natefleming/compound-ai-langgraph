from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod
from functools import partial


from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from guardrails.guard import Guard

from app.router import filter_out_routes
from app.messages import add_name, get_last_human_message
from app.tools import (
    create_genie_tool, 
    create_vector_search_tool, 
    create_unity_catalog_tools, 
    create_router_tool
)

from app.prompts import (
    genie_prompt, 
    vector_search_agent_prompt, 
    vector_search_chain_prompt,
    unity_catalog_prompt, 
    router_prompt
)
from app.retrievers import create_vector_search_retriever

class AgentBase(ABC):
    """Abstract base class for agents."""

    def __init__(self) -> None:
        """Initializes the base agent."""
        self._next: Optional[Agent] = None

    @abstractmethod
    def as_runnable() -> RunnableSequence:
        """Abstract method to convert the agent to a runnable sequence."""
        ...

    def then(self, next: 'AgentBase') -> 'AgentBase':
        """Sets the next agent in the sequence.

        Args:
            next (AgentBase): The next agent to be executed.

        Returns:
            AgentBase: The next agent.
        """
        self._next = next
        return next


class Agent(AgentBase):
    """Concrete implementation of an agent."""

    def __init__(
        self,
        name: str,
        topics: str,
        llm: BaseChatModel,
        prompt: Optional[str] = None,
        tools: List[Tool] = [],
        post_guard: Optional[Guard] = None
    ) -> None:
        """Initializes the agent.

        Args:
            name (str): The name of the agent.
            topics (str): The topics of the agent.
            llm (BaseChatModel): The language model to be used by the agent.
            prompt (Optional[str], optional): The prompt for the agent. Defaults to None.
            tools (List[Tool], optional): The tools to be used by the agent. Defaults to [].
            post_guard (Optional[Guard], optional): The guard to be used after execution. Defaults to None.
        """
        self.name = name
        self.topics = topics
        self.llm = llm
        self.prompt = prompt
        self.tools = tools
        self.post_guard = post_guard
        if self.post_guard is not None:
            self.post_guard.name = f"{name}-{post_guard.name}"

    def as_runnable(self) -> RunnableSequence:
        """Converts the agent to a runnable sequence.

        Returns:
            RunnableSequence: The runnable sequence.
        """

        chain: RunnableSequence = (
            RunnableLambda(filter_out_routes) |
            RunnableLambda(self._get_messages) |
            self.llm.bind_tools(self.tools) |
            partial(add_name, name=self.name)
        )

        return chain
    
    def _get_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Prepends the prompt to the messages if it exists.

        Args:
            messages (List[BaseMessage]): The list of messages.

        Returns:
            List[BaseMessage]: The modified list of messages.
        """
        result_messages: List[BaseMessage]
        if self.prompt is not None:
            result_messages = [SystemMessage(content=self.prompt)] + messages
        else:
            result_messages = messages

        return result_messages



def create_agent(
    name: str,
    topics: str,
    llm: BaseChatModel,
    prompt: Optional[str] = None,
    tools: List[Tool] = [],
    post_guard: Optional[Guard] = None
) -> Agent:
    """Creates an agent.

    Args:
        name (str): The name of the agent.
        topics (str): The topics of the agent.
        llm (BaseChatModel): The language model to be used by the agent.
        prompt (Optional[str], optional): The prompt for the agent. Defaults to None.
        tools (List[Tool], optional): The tools to be used by the agent. Defaults to [].
        post_guard (Optional[Guard], optional): The guard to be used after execution. Defaults to None.

    Returns:
        Agent: The created agent.
    """
    return Agent(
        name=name, 
        topics=topics, 
        llm=llm, 
        prompt=prompt, 
        tools=tools,
        post_guard=post_guard
    )

def create_vector_search_chain(
    llm: BaseChatModel,
    endpoint_name: str,
    index_name: str,
    primary_key: str,
    text_column: str,
    doc_uri: str,
    columns: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    name: Optional[str] = "vector_search_chain",
    topics: Optional[str] = "process, guides, technical documentation, how-to"
) -> Agent:
    
    prompt_template: str = vector_search_chain_prompt()
   
    prompt: PromptTemplate = PromptTemplate(
        #input_variables=["context", "question"],
        input_variables=["summaries", "question"],
        template=prompt_template
    )

    vector_search_as_retriever: VectorStoreRetriever = create_vector_search_retriever(
        endpoint_name=endpoint_name,
        index_name=index_name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        columns=columns,
        parameters=parameters,
    )

    qa_chain: RunnableSequence = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=vector_search_as_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    class _Agent(Agent):

        def __init__(self) -> None:
            super().__init__(name=name, topics=topics, llm=llm)

        def as_runnable(self) -> RunnableSequence:
            """Converts the agent to a runnable sequence.

            Returns:
                RunnableSequence: The runnable sequence.
            """

            def run_qa_chain(messages: List[BaseMessage]) -> BaseMessage:
                message: HumanMessage = get_last_human_message(messages)
                if not message:
                    raise ValueError("No Human Message found in messages")

                question: str = message.content
                result: Dict[str, Any] = qa_chain.invoke({"question": question})
                answer: str = result["answer"]
                sources: str = result["sources"]

                return AIMessage(content=answer)
            
            chain: RunnableSequence = (
                RunnableLambda(filter_out_routes) |
                RunnableLambda(run_qa_chain) |
                partial(add_name, name=self.name)
            )


            return chain
         

    agent: Agent = _Agent()

    return agent


def create_router_agent(
    llm: BaseChatModel,
    agents: List[Agent],
    default_agent: Optional[Agent] = None,
    name: str = "router",
    topics: str = "Classifying and routing user prompts"
) -> Agent:


    router_tool: Tool = create_router_tool(choices=agents)
    
    prompt: str = router_prompt(agents=agents, default_agent=default_agent)

    router_agent: Agent = create_agent(
        name=name, topics=topics, llm=llm, prompt=prompt, tools=[router_tool]
    )

    return router_agent

        
def create_unity_catalog_agent(
    llm: BaseChatModel,
    warehouse_id: str,
    functions: List[str],
    name: Optional[str] = "unity_catalog",
    topics: Optional[str] = "Unity Catalog tools and functions."
) -> Agent:
    """Creates a Unity Catalog agent.

    Args:
        llm (BaseChatModel): The language model to be used by the agent.
        warehouse_id (str): The warehouse ID.
        functions (List[str]): The list of functions.
        name (Optional[str], optional): The name of the agent. Defaults to "unity_catalog".
        topics (Optional[str], optional): The topics of the agent. Defaults to "Answer questions using Unity Catalog tools and functions.".

    Returns:
        Agent: The created Unity Catalog agent.
    """
    unity_catalog_tools: List[Tool] = create_unity_catalog_tools(
        warehouse_id=warehouse_id,
        functions=functions,
    )
    prompt: str = unity_catalog_prompt()

    unity_catalog_agent: Agent = create_agent(
        name=name,
        topics=topics,
        llm=llm,
        prompt=prompt,
        tools=unity_catalog_tools
    )

    return unity_catalog_agent


def create_genie_agent(
    llm: BaseChatModel,
    space_id: str,
    workspace_host: Optional[str] = None,
    token: Optional[str] = None,
    name: Optional[str] = "genie",
    topics: Optional[str] = "Databricks Genie tools."
) -> Agent:
    """Creates a Genie agent.

    Args:
        llm (BaseChatModel): The language model to be used by the agent.
        space_id (str): The space ID.
        workspace_host (Optional[str], optional): The workspace host. Defaults to None.
        token (Optional[str], optional): The token. Defaults to None.
        name (Optional[str], optional): The name of the agent. Defaults to "genie".
        topics (Optional[str], optional): The topics of the agent. Defaults to "Answer questions using Databricks Genie tools.".
    Returns:
        Agent: The created Genie agent.
    """
    genie_tool: Tool = create_genie_tool(
        space_id=space_id,
        workspace_host=workspace_host,
        token=token
    )
    prompt: str = genie_prompt(genie_tool.name)

    genie_agent: Agent = create_agent(
        name=name,
        topics=topics,
        llm=llm,
        prompt=prompt,
        tools=[genie_tool]
    )

    return genie_agent



def create_vector_search_agent(
    llm: BaseChatModel,
    endpoint_name: str,
    index_name: str,
    primary_key: str,
    text_column: str,
    doc_uri: str,
    columns: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    name: Optional[str] = "vector_search",
    topics: Optional[str] = "process, guides, technical documentation, how-to"
) -> Agent:
    """Creates a Vector Search agent.

    Args:
        llm (BaseChatModel): The language model to be used by the agent.
        endpoint_name (str): The endpoint name.
        index_name (str): The index name.
        primary_key (str): The primary key.
        text_column (str): The text column.
        doc_uri (str): The doc_uri.
        columns (List[str], optional): The list of columns. Defaults to None.
        parameters (Dict[str, Any], optional): The parameters. Defaults to None.
        name (Optional[str], optional): The name of the agent. Defaults to "vector_search".
        topics (Optional[str], optional): The topics of the agent. Defaults to "Answer questions about Databricks".
    Returns:
        Agent: The created Vector Search agent.
    """
    vector_search_tool: Tool = create_vector_search_tool(
        endpoint_name=endpoint_name,
        index_name=index_name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        columns=columns,
        parameters=parameters,
    )

    prompt: str = vector_search_agent_prompt(tool_name=vector_search_tool.name)

    vector_search_agent: Agent = create_agent(
        name=name,
        topics=topics,
        llm=llm,
        prompt=prompt,
        tools=[vector_search_tool]
    )

    return vector_search_agent