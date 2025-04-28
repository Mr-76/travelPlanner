from typing_extensions import TypedDict
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from io import BytesIO
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage, RemoveMessage
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
import random
from typing import Literal


class State(TypedDict):
    graph_state: str

def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}


def decide_mood(state) -> Literal["node_2", "node_3","node_1"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() > 0.5 and random.random() < 0.6:
        return "node_2"
    elif random.random() < 0.5:
        return "node_1"
    else:
        return "node_3"



builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)



graph = builder.compile()

img_bytes = graph.get_graph().draw_mermaid_png()
img = mpimg.imread(BytesIO(img_bytes), format='png')
plt.imshow(img)
plt.axis('off')
plt.show()

print("GRAPH READY")

graph.invoke({"graph_state" : "Hi, this is Lance."})
