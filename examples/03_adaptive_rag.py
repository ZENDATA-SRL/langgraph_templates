# Adaptive RAG is a strategy for RAG that unites
# - (1) query analysis with
# - (2) active / self-corrective RAG.

# Query analysis to route across:
# - No Retrieval
# - Single-shot RAG
# - Iterative RAG
from typing import Annotated, List, Literal

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun


## Router
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# LLM with function call
llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0.5, "max_tokens": 2048},
)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
question = "Who will the Bears draft first in the NFL draft?"
print(
    f"ROUTER with question: {question} \n",
    question_router.invoke({"question": question}),
)
print(
    "ROUTER with question: What are the types of agent memory?\n",
    question_router.invoke({"question": "What are the types of agent memory?"}),
)


### Retrieval Grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
doc_txt = "Agent memory is the ability of an agent to remember past experiences and use them to make decisions."
print(
    f"GRADER with question: {question}, documents: {doc_txt}",
    retrieval_grader.invoke({"question": question, "document": doc_txt}),
)


### Hallucination Grader
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
generation = "Agent memory is the agent dream when it sleeps."
hallucination_grader = hallucination_prompt | structured_llm_grader
print(
    f"ALLUCINATION GRADER with documents: {doc_txt}, generation: {generation}",
    hallucination_grader.invoke({"documents": doc_txt, "generation": generation}),
)


### Answer Grader
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)
# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader = answer_prompt | structured_llm_grader
print(
    f"ANSWER GRADER with question: {question}, generation: {generation}",
    answer_grader.invoke({"question": question, "generation": generation}),
)

### Question Re-writer
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
print(
    f"QUESTION REWRITER with question: {question}",
    question_rewriter.invoke({"question": question}),
)

web_search_tool = DuckDuckGoSearchRun(max_results=3)


# Create the Graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]


# Nodes
def web_search(state: GraphState):
    # Search the web for the question
    print("---WEB SEARCH---")
    question = state["question"]
    documents = web_search_tool.invoke(question)
    return {"documents": documents}

def retrieve(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    # Retrieval
    documents = [
        "Agents memory is for remembering past experiences.",
        "Agents memory has the previous conversation history",
        "Agents memory contains both long and short memory",
    ]
    return {"documents": documents, "question": question}


def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    system = """You are an LLM generating an answer to a user question given a context.
    Provide your answer based on this context:
    {context}
    \n"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    rag_chain = prompt | llm

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation.content}

def grade_retrieval(state: GraphState):
    # Grade the retrieved documents
    print("---GRADE RETRIEVAL---")
    question = state["question"]
    documents = state["documents"]
    filtered_documents = []
    for doc in documents:
        grade: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc})
        print(f"Document: {doc} Grade: {grade.binary_score}")
        if grade.binary_score == "yes":
            filtered_documents.append(doc)
    return {"grade": len(filtered_documents) > 0, "documents": filtered_documents}

def grade_answer(state: GraphState):
    # Grade the generated answer
    print("---GRADE ANSWER---")
    question = state["question"]
    generation = state["generation"]
    grade: GradeAnswer = answer_grader.invoke({"question": question, "generation": generation})
    print(f"Generation: {generation} Grade: {grade.binary_score}")
    return {"grade": grade.binary_score == "yes"}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

# Edge functions
def route_query(state: GraphState):
    # Route the question to the most relevant datasource
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
def decide_to_generate(state: GraphState):
    # Decide whether to generate an answer
    print("---DECIDE TO GENERATE---")
    state["question"]
    filtered_documents = state["documents"]
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
def grade_generation_v_documents_and_question(state: GraphState):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score: GradeHallucinations = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score
    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
graph_builder = StateGraph(GraphState)

# Add nodes
graph_builder.add_node("route_query", route_query)
