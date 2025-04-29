from IPython.display import Image, display
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from io import BytesIO

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

#this code can loop asking the user for input to multiply values also can keep track and multiply re reuslt valiues of other tools calls eve wihtou memory
#need to obs why can doit without memory.
#obs the result node implicity as it seems is called as a conditional edge...

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b
@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

@tool
def decide_exit(a: str) -> bool:
    """Returns True or Flase.

    Args:
        a: Boolean 
    """
    if a == "False":
        print("its false not exiting")
        return False
    else:
        print("its not false continue")
        return True



ollama_url = "http://192.168.0.12:11434"
llm = ChatOllama(
    #model="llama3.2:3b",        # Your custom model name
    model="mistral-nemo:latest",        # Your custom model name
    base_url=ollama_url,        # Ollama running on remote/local IP
)
tools = [add, multiply, divide, decide_exit]
llm_with_tools = llm.bind_tools(tools)


# System message
sys_msg = SystemMessage(content="""You are a helpful assistant tasked with performing arithmetic on a set of inputs., beware that the user can keep asking ou to do more operations 
                        and you should not take in consideration the old messages to do the current arithmetic""")

# Node
def assistant(state: MessagesState):
    input1 = input("Describe what operatin you want to do mult, sum division....")
    statenew = state["messages"] + [HumanMessage(content=input1)]
    return {"messages": [llm_with_tools.invoke([sys_msg] + statenew)]}


def result_node(state: MessagesState):
    print("resulting.........")
    print(state["messages"])


def decision_node(state: MessagesState):
    input1 = input("Do you want to exit? Yes or No: ")
    continue1 = llm_with_tools.invoke([SystemMessage(content="If the user says stop, i wanna exit, i wanna stop or anything negative use the argument a to be of value True and if the user wants to continue using whci means using words as i wanna continue, please continue , go on and other words you need to return the value o a as False"),HumanMessage(content=input1)])
    print(continue1)
    a_value = continue1.tool_calls[0]['args']['a']
    print(f"Decision made by tool: a = {a_value}")

    if a_value == "True":
        return "END"
    return "assistant"
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("result_node", result_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_edge("assistant", "result_node")
builder.add_conditional_edges("result_node",decision_node)

builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)


react_graph = builder.compile()

img_bytes = react_graph.get_graph().draw_mermaid_png()
img = mpimg.imread(BytesIO(img_bytes), format='png')
plt.imshow(img)
plt.axis('off')
plt.show()

messages = [HumanMessage(content="")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()


















