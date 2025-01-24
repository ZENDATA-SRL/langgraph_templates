# A Zero Shot Agent is an Agentic architecture where an LLM has some tools (python functions) that it can call and exploits the tools result to answer a user question.
# The agent uses a reflection mechanism to call the tools and get the result.

# Start building the graph state the llm engine and the tools
import json
from typing import Annotated

from langchain_aws import ChatBedrock
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pymongo import AsyncMongoClient
from typing_extensions import TypedDict


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

# Create the graph builder
graph_builder = StateGraph(State)

# Add the tools to the graph
tools = [TavilySearchResults(max_results=2)]

llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0.5, "max_tokens": 2048},
)
# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

# Next we need to create a function to actually run the tools if they are called. We'll do this by adding the tools to a new node.

# If you want to create custom tool this is the template to follow
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# With the tool node added, we can define the conditional_edges.
# Edges route the control flow from one node to the next. 
# Conditional edges usually contain "if" statements to route to different nodes depending on the current graph state. These functions receive the current graph state and return a string or list of strings indicating which node(s) to call next.
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# Let's run the graph
# No tool call
for event in graph.stream({"messages": [{"role": "user", "content": "Hello World"}]}):
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)
        
        
# Tool call
for event in graph.stream({"messages": [{"role": "user", "content": "I want to search the langgrah documentation"}]}):
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)
        

"""This chatbot can answer user questions, but it doesn't remember the context of previous interactions.
LangGraph solves this problem through persistent checkpointing. 
If you provide a checkpointer when compiling the graph and a thread_id when calling your graph, LangGraph automatically saves the state after each step. 
When you invoke the graph again using the same thread_id, the graph loads its saved state, allowing the chatbot to pick up where it left off."""
MONGODB_URI="mongodb://localhost:27017"
mongodb_client = AsyncMongoClient(MONGODB_URI)
checkpointer = AsyncMongoDBSaver(mongodb_client)
config = {"configurable": {"thread_id": "2"}} 
response = graph.invoke({"messages": [("user", "What's the weather in sf?")]}, config)

"""The Graph state can be customized in order to manage complex behaviour based on the execution of some nodes.
Let's have the chatbot research the birthday of an entity. We will add name and birthday keys to the state.
Adding this information to the state makes it easily accessible by other graph nodes.
"""
@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)