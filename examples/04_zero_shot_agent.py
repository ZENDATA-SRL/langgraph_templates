# Agents can handle sophisticated tasks, but their implementation is often straightforward. 
# They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully.
# The agent uses a reflection mechanism to call the tools and get the result.

# Start building the graph state the llm engine and the tools
import json
import os
from datetime import date, datetime
from typing import Annotated, Optional

from langchain_aws import ChatBedrock
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pymongo import MongoClient
from typing_extensions import TypedDict


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# Create the graph builder
graph_builder = StateGraph(State)


# Create the tools
@tool
def fetch_user_flight_information(config: RunnableConfig) -> list[dict]:
    """Fetch all tickets for the user along with corresponding flight information and seat assignments."""
    ## Tool implementation would go here
    example_result = [
        {"ticket_no": "1234", "flight_id": 1, "seat": "A1"},
        {"ticket_no": "5678", "flight_id": 2, "seat": "B2"},
    ]
    return example_result


@tool
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    start_time: Optional[date | datetime] = None,
    end_time: Optional[date | datetime] = None,
    limit: int = 20,
) -> list[dict]:
    """Search for flights based on departure airport, arrival airport, and departure time range."""
    ## Tool implementation would go here
    example_result = [
        {
            "FCO": "Fiumicino",
            "JFK": "John F. Kennedy",
            "departure_time": "2022-01-01 08:00:00",
            "arrival_time": "2022-01-01 12:00:00",
        },
    ]
    return example_result


@tool
def update_ticket_to_new_flight(
    ticket_no: str, new_flight_id: int, *, config: RunnableConfig
) -> str:
    """Update the user's ticket to a new valid flight."""
    ## Tool implementation would go here
    return "Ticket successfully updated to new flight."


@tool
def cancel_ticket(ticket_no: str, *, config: RunnableConfig) -> str:
    """Cancel the user's ticket and remove it from the database."""
    ## Tool implementation would go here
    return "Ticket successfully cancelled."


# Let's create the Agent node #
"""Next, define the assistant function. This function takes the graph state, formats it into a prompt, and then calls an LLM for it to predict the best response."""
llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0.5, "max_tokens": 2048},
)


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

tools = [
    DuckDuckGoSearchRun(max_results=1),
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket,
]
assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools=tools)


# Let's define some helper functions to handle tools errors and wrap the tools functions in a ToolNode
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# Define nodes: these do the work
graph_builder.add_node("assistant", Assistant(assistant_runnable))
graph_builder.add_node("tools", create_tool_node_with_fallback(tools))
# Define edges: these determine how the control flow moves
graph_builder.add_edge(START, "assistant")
graph_builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
graph_builder.add_edge("tools", "assistant")


"""This chatbot can answer user questions, but it doesn't remember the context of previous interactions.
LangGraph solves this problem through persistent checkpointing. 
If you provide a checkpointer when compiling the graph and a thread_id when calling your graph, LangGraph automatically saves the state after each step. 
When you invoke the graph again using the same thread_id, the graph loads its saved state, allowing the chatbot to pick up where it left off."""
MONGODB_URI = os.environ["MONGODB_URI"]
mongodb_client = MongoClient(MONGODB_URI)
checkpointer = MongoDBSaver(mongodb_client)
config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",  # We can pass extra parameters to a graph using both state and run configuration
        # Checkpoints are accessed by thread_id
        "thread_id": "6",
    }
}
graph = graph_builder.compile(checkpointer=checkpointer)
while True:
    question = input("Ask a question: ")
    response = graph.invoke({"messages": [("user", question)]}, config)
    print(response["messages"][-1].content)
