from typing_extensions import TypedDict
from io import BytesIO
from IPython.display import Image, display
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from langgraph.graph import StateGraph, START, END




class OverallState(TypedDict):
    foo: int

class PrivateState(TypedDict):
    baz: int

def node_1(state: OverallState) -> PrivateState:
    print("---Node 1---")
    return {"baz": state['foo'] + 1}

def node_2(state: PrivateState) -> OverallState:
    print("---Node 2---")
    input1 = input("give me a number")
    return {"foo": state['baz'] + int(input1)}

def should_continue(state:OverallState) -> ["node_1","exit_node"]:
    continueConversation = input("want to sum more ?")
    if continueConversation == "Y":
        return "node_1"
    else:
        return "exit_node"


def exit_node(state: OverallState):
    print("---exitingnode---")


# Build graph
builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("exit_node",exit_node )

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_conditional_edges("node_2",should_continue)
builder.add_edge("exit_node", END)
graph = builder.compile()

img_bytes = graph.get_graph().draw_mermaid_png()
img = mpimg.imread(BytesIO(img_bytes), format='png')
plt.imshow(img)
plt.axis('off')
plt.show()

result = graph.invoke({"foo":1})
print(result)
