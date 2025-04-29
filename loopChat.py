from IPython.display import Image, display
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from io import BytesIO


from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState

# LLM
ollama_url = "http://192.168.0.12:11434"
model = ChatOllama(
    model="llama3.2:3b",        # Your custom model name
    base_url=ollama_url,        # Ollama running on remote/local IP
)

#this programs makes a working chat loop but output its not very good

# State
class State(MessagesState):
    summary: str

# Define the logic to call the model


def call_model(state: State, config: RunnableConfig):
    # Get summary if it exists
    summary = state.get("summary", "")
    input4 = input("ask me anything...")
    messages1 = state["messages"] + [HumanMessage(content=input4)]
    # If there is summary, then we add it
    if summary:
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + messages1
    else:
        messages = messages1
    response = model.invoke(messages, config)
    return {"messages": response}


def summarize_conversation(state: State):
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Determine whether to end or summarize the conversation




def should_continue(state: State) -> ["conversation", "END"]:
    """Return the next node to execute."""
    print(state["messages"])
    input2 = input("You the user you want to continue ?")

    if input2 == "Y":
        return "conversation"
    else:
        return END

    # Otherwise we can just end

# Define a new graph


workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_edge("conversation", "summarize_conversation")
workflow.add_conditional_edges("summarize_conversation", should_continue)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))

# Create a thread
config = {"configurable": {"thread_id": "1"}}


img_bytes = graph.get_graph().draw_mermaid_png()
img = mpimg.imread(BytesIO(img_bytes), format='png')
plt.imshow(img)
plt.axis('off')
plt.show()

# Start conversation
result =  graph.invoke({"messages": [HumanMessage(content="hi! I'm Lance")]},config)
print(result)
#for chunk in graph.stream({}, config, stream_mode="updates"):
#    print(chunk)
