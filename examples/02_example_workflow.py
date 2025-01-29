# With Langgraph it is possible to create LLM workflows or Agents architecture.
# Workflows are systems where LLMs and tools are orchestrated through predefined code paths.
# Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

# In this script we provide some sample workflows that can be used to orchestrate LLMs and tools in a predefined code path.
import operator
from typing import Annotated, List
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langgraph.constants import Send

# Example Workflow 1: Prompt Chaining
"""Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one.
You can add programmatic checks on any intermediate steps to ensure that the process is still on track.

When to use this workflow: 
This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. 
The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.
"""

# Graph state
class ChainState(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0.5, "max_tokens": 2048},
)

# Nodes
def generate_joke(state: ChainState):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


def check_punchline(state: ChainState):
    """Gate function to check if the joke has a punchline"""
    # Simple check - does the joke contain "?" or "!"
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Fail"
    return "Pass"


def improve_joke(state: ChainState):
    """Second LLM call to improve the joke"""

    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: ChainState):
    """Third LLM call for final polish"""

    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}


# Build workflow
workflow = StateGraph(ChainState)

# Add nodes
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

# Compile and invoke the graph
graph = workflow.compile()
state = graph.invoke({"topic": "cats"})
print("CHAIN\n", state["final_joke"])

# Example Workflow 2: Parallel Processing
"""LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. 
This workflow, parallelization, manifests in two key variations: 
    - Sectioning: Breaking a task into independent subtasks run in parallel. 
    - Voting: Running the same task multiple times to get diverse outputs.
    
When to use this workflow: 
Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. 
For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect"""
    
# Graph state
class ParallelState(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


# Nodes
def call_llm_1(state: ParallelState):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def call_llm_2(state: ParallelState):
    """Second LLM call to generate story"""

    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}


def call_llm_3(state: ParallelState):
    """Third LLM call to generate poem"""

    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}


def aggregator(state: ParallelState):
    """Combine the joke and story into a single output"""

    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


# Build workflow
parallel_graph_builder = StateGraph(ParallelState)

# Add nodes
parallel_graph_builder.add_node("call_llm_1", call_llm_1)
parallel_graph_builder.add_node("call_llm_2", call_llm_2)
parallel_graph_builder.add_node("call_llm_3", call_llm_3)
parallel_graph_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes
parallel_graph_builder.add_edge(START, "call_llm_1")
parallel_graph_builder.add_edge(START, "call_llm_2")
parallel_graph_builder.add_edge(START, "call_llm_3")
parallel_graph_builder.add_edge("call_llm_1", "aggregator")
parallel_graph_builder.add_edge("call_llm_2", "aggregator")
parallel_graph_builder.add_edge("call_llm_3", "aggregator")
parallel_graph_builder.add_edge("aggregator", END)
parallel_workflow = parallel_graph_builder.compile()

# Invoke
state = parallel_workflow.invoke({"topic": "cats"})
print("PARALLEL\n", state["combined_output"])

# Example Workflow 3: Routing 
"""Routing classifies an input and directs it to a specialized followup task. 
This workflow allows for separation of concerns, and building more specialized prompts. 
Without this workflow, optimizing for one kind of input can hurt performance on other inputs.

When to use this workflow: 
Routing works well for complex tasks where there are distinct categories that are better handled separately, 
and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm."""
# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="The next step in the routing process"
    )

# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)

# State
class RouteState(TypedDict):
    input: str
    decision: str
    output: str

# Nodes
def llm_call_1(state: RouteState):
    """Write a story"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_2(state: RouteState):
    """Write a joke"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_3(state: RouteState):
    """Write a poem"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_router(state: RouteState):
    """Route the input to the appropriate node"""
    # Run the augmented LLM with structured output to serve as routing logic
    decision: Route = router.invoke(
        [
            SystemMessage(
                content="Route the input to story, joke, or poem based on the user's request."
            ),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}


# Conditional edge function to route to the appropriate node
def route_decision(state: RouteState):
    # Return the node name you want to visit next
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"


# Build workflow
router_graph_builder = StateGraph(RouteState)

# Add nodes
router_graph_builder.add_node("llm_call_1", llm_call_1)
router_graph_builder.add_node("llm_call_2", llm_call_2)
router_graph_builder.add_node("llm_call_3", llm_call_3)
router_graph_builder.add_node("llm_call_router", llm_call_router)

# Add edges to connect nodes
router_graph_builder.add_edge(START, "llm_call_router")
router_graph_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_graph_builder.add_edge("llm_call_1", END)
router_graph_builder.add_edge("llm_call_2", END)
router_graph_builder.add_edge("llm_call_3", END)

# Compile workflow and Invoke
router_workflow = router_graph_builder.compile()
state = router_workflow.invoke({"input": "Write me a joke about cats"})
print("ROUTER\n", state["output"])

# Example Workflow 4: Orchestrator-Worker
"""In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

When to use this workflow: 
This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task).
Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.
"""
# Schema for structured output to use in planning
class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections)

# Graph state
class State(TypedDict):
    topic: str  # Report topic
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_report: str  # Final report


# Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


# Nodes
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""

    # Generate queries
    report_sections: Sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}"),
        ]
    )

    return {"sections": report_sections.sections}


def llm_call(state: WorkerState):
    """Worker writes a section of the report"""
    # Generate section
    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            ),
        ]
    )
    # Write the updated section to completed sections
    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """Synthesize full report from sections"""
    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""
    # Kick off section writing in parallel via Send() API
    print(f"Assigning workers to write {len(state['sections'])} sections.")
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


# Build workflow
orchestrator_worker_graph_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_graph_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_graph_builder.add_node("llm_call", llm_call)
orchestrator_worker_graph_builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
orchestrator_worker_graph_builder.add_edge(START, "orchestrator")
orchestrator_worker_graph_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_graph_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_graph_builder.add_edge("synthesizer", END)

# Compile the workflow and invoke
orchestrator_worker = orchestrator_worker_graph_builder.compile()
state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})
print("ORCHESTRATOR WORKER\n", state["final_report"])

# Example Workflow 5: Evaluator-optimizer
"""In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.
When to use this workflow: 
This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value.
The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback."""
# Graph state
class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )

# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)

# Nodes
def llm_call_generator(state: State):
    """LLM generates a joke"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""
    grade: Feedback = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

# Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


# Build workflow
optimizer_graph_builder = StateGraph(State)

# Add the nodes
optimizer_graph_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_graph_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
optimizer_graph_builder.add_edge(START, "llm_call_generator")
optimizer_graph_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_graph_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # Name returned by route_joke : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow and invoke
optimizer_workflow = optimizer_graph_builder.compile()
state = optimizer_workflow.invoke({"topic": "Cats"})
print("EVALUATOR OPTIMIZER\n", state["joke"])