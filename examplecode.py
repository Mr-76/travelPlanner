# Import necessary libraries
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
from langgraph.checkpoint.sqlite import SqliteSaver

#from https://blog.gopenai.com/building-an-event-planner-agent-using-tavily-lang-graph-and-openai-1597553fb3d1
# Load environment variables
load_dotenv()

# Set up the in-memory checkpoint storage for agent states
memory = SqliteSaver.from_conn_string(":memory:")

# Define the agent state structure
class AgentState(TypedDict):
    task: str
    venue_plan: str
    logistic_plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Initialize the OpenAI GPT model (GPT-4o)
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Define the query structure for the researcher agent
class Queries(BaseModel):
    queries: List[str]

# Set up the Tavily search client
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Define the venue planner node
VENUE_PLAN_PROMPT = """You are an expert venue planner tasked with planning a 
high level event. Find a venue that meets criteria for an event. expected 
output is all the details of a specifically chosen venue you found to 
accommodate the event. With a keen sense of space and understanding of 
event logistics, you excel at finding and securing the perfect venue that
fits the event's theme, size, and budget constraints."""

def venue_plan_node(state: AgentState):
    messages = [
        SystemMessage(content=VENUE_PLAN_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"venue_plan": response.content}

# Define the logistic manager node
LOGISTIC_PROMPT = """You are a logistic manager. your goal is to Manage all 
logistics for the event, including catering and equipment. You need to be 
organized and detail-oriented, ensuring every logistical aspect of the 
event from catering to equipment setup is flawlessly executed."""

def logistic_manage_node(state: AgentState):
    messages = [
        SystemMessage(content=LOGISTIC_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"logistic_plan": response.content}

# Define the research plan node
RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing 
information that can be used when planning the following plan. 
Generate a list of search queries that will gather 
any relevant information. Only generate 3 queries max."""

def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

# Define the planner node
PLANNER_PROMPT = """You are an event planning assistant tasked with planning 
an event. Generate the best plan possible for the user's request and the 
initial outline. If the user provides critique, respond with a revised 
version of your previous attempts."""

def planner_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']} \n\n Here is venue plan:\n\n{state['venue_plan']}  \n\n Here is logistic plan:\n\n{state['logistic_plan']}")
    messages = [
        SystemMessage(content=PLANNER_PROMPT.format(content=content)),
        user_message
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

# Define the editor node
EDITOR_PROMPT = """You are an editor grading an essay submission. 
Generate critique and recommendations for the user's submission. 
Provide detailed recommendations, including requests for length, depth, style, etc."""

def editor_node(state: AgentState):
    messages = [
        SystemMessage(content=EDITOR_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

# Define the research critique node
RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing 
information that can be used when making any requested revisions. 
Generate a list of search queries that will gather any 
relevant information. Only generate 3 queries max."""

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

# Define a function to check whether the process should continue
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

# Build the agent state graph and define the flow of nodes
builder = StateGraph(AgentState)

# Adding nodes to the graph
builder.add_node("venue_planner", venue_plan_node)
builder.add_node("logistic_manager", logistic_manage_node)
builder.add_node("generate", planner_node)
builder.add_node("editor", editor_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

# Set the entry point to the venue planner
builder.set_entry_point("venue_planner")

# Add conditional edge to check if the process should end or continue
builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "editor": "editor"}
)

# Define the flow of the nodes with edges
builder.add_edge("venue_planner", "logistic_manager")
builder.add_edge("logistic_manager", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("editor", "research_critique")
builder.add_edge("research_critique", "generate")

# Compile the graph
graph = builder.compile(checkpointer=memory)

# Streaming the response based on user input
thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "plan for an open ground musical event in Colombo, Sri Lanka",
    "max_revisions": 2,
    "revision_number": 1,
}, thread):

    print(s)

