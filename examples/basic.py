# Langraph General Concepts
# LangGraph is a stateful orchestration framework designed to enhance control over agent workflows, particularly in applications involving large language models (LLMs)

"""A Langraph Graph builder needs always a StateGraph to build a graph. A StateGraph object defines the structure of our chatbot as a "state machine".
We'll add nodes to represent the llm and functions our chatbot can call and edges to specify how the bot should transition between these functions.
    1. The nodes in the grah can receive the current state as input and can output an update to the state.
    2. Updates to messages will be appended to the existing list rather than overwriting it, thanks to the prebuilt add_messages function used with the Annotated syntax.
The start and end are special nodes that represent the beginning and end of the conversation. (imported from langgraph.graph)"""

# <Example>
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pymongo import AsyncMongoClient
from typing_extensions import TypedDict

load_dotenv()

"""NOTE: The Graph state can be customized in order to manage complex behaviour based on the execution of some nodes."""
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    # Add more state keys here as needed


async def main():
    graph_builder = StateGraph(State)
    # Next, add a "chatbot" node. Nodes represent units of work. They are typically regular python functions.

    llm = ChatBedrock(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={"temperature": 0.5, "max_tokens": 2048},
    )

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    """The first argument is the unique node name while the second argument is the function or object that will be called whenever the node is used.
    Notice how the chatbot node function takes the current State as input and returns a dictionary containing an updated messages list under the key "messages".
    This is the basic pattern for all LangGraph node functions."""
    graph_builder.add_node("chatbot", chatbot)

    """The add_messages function in our State will append the llm's response messages to whatever messages are already in the state.
    Next, add an entry point. This tells our graph where to start its work each time we run it."""
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # Finally, we'll want to be able to run our graph. To do so, call "compile()" on the graph builder. This creates a "CompiledGraph" we can use invoke on our state.
    graph = graph_builder.compile()

    # We can call the graph with an initial state to run it. With stream we got the streaming output of the Llm.
    for event in graph.stream(
        {"messages": [{"role": "user", "content": "Hello World"}]}
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

    """This chatbot can answer user questions, but it doesn't remember the context of previous interactions.
    LangGraph solves this problem through persistent checkpointing. 
    If you provide a checkpointer when compiling the graph and a thread_id when calling your graph, LangGraph automatically saves the state after each step. 
    When you invoke the graph again using the same thread_id, the graph loads its saved state, allowing the chatbot to pick up where it left off.
    """
    mongodb_client = AsyncMongoClient(os.environ["MONGODB_URI"])
    checkpointer = AsyncMongoDBSaver(mongodb_client)
    graph = graph_builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "2"}}
    response = await graph.ainvoke(
        {"messages": [("user", "Hello my name is Mario")]}, config
    )
    print(response["messages"][-1].content)
    response = await graph.ainvoke({"messages": [("user", "What is my name?")]}, config)
    print(response["messages"][-1].content)

    latest_checkpoint = await checkpointer.aget_tuple(config)
    print(latest_checkpoint)
    # </Example>


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
