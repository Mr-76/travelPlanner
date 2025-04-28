from langgraph.graph import StateGraph, END
import matplotlib.image as mpimg

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from io import BytesIO
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel
from dotenv import load_dotenv
from tavily import TavilyClient
from IPython.display import Image, display
from typing import Literal, TypedDict, List
import operator

#prob use tool caling to decide either ot exit or not from the conversation
#all so every call to the graph run it all, but u can keep track with using MessagesState in you sate
#sou can save the messages, and every call to the graph will have the messages from before


memory = MemorySaver()

ollama_url = "http://192.168.0.12:11434"
llm = ChatOllama(
    model="llama3.2:3b",        # Your custom model name
    base_url=ollama_url,        # Ollama running on remote/local IP
)


sys_msg = SystemMessage(content="""You are a helpful assistant that will give recomandations and talk about traveling
                        you will give the client good optins depeing on where they want to go based and what you know
                        alredy about that place, so u can give for example a list of places to where the person can go
                        also with recomandations and explaining each one of the places and why would be good,
                        also when the cliend states that it wants to end you shoud end the conversation and keep this words,
                        goodbye see you no more questios , ending, stop ,bye, have a nice day , that`s ll nothing else, no futher tasks, we are done""")

class StateTravelTalk(MessagesState):
    continue_talk: bool
    summary: str



def call_model(state: StateTravelTalk, config: RunnableConfig):
    summary = state.get("summary", "")
    if summary:
        system_message = SystemMessage(content=f"Summary of conversation earlier: {summary}")
        messages = [system_message] + state["messages"]
    else:
        messages = state["messages"]

    response = llm.invoke([sys_msg] + messages)

    return {"messages": state["messages"] + [response]}


def summarize_conversation(state: StateTravelTalk):
    summary = state.get("summary", "")

    if summary:
        prompt = f"This is summary of conversation to date: {summary}\n\nExtend it by including new conversation."
    else:
        prompt = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=prompt)]

    response = llm.invoke([sys_msg] + messages)

    # Delete old messages if needed (optional)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return {
        "summary": response.content,
        "messages": delete_messages + state["messages"][-2:],  # Keep last few messages
    }


def is_ending_conversation(messages):
    """
    Check if the assistant said something that indicates the conversation should end.
    """
    if not messages:
        return False

    last_message = messages[-1]

    if hasattr(last_message, 'content') and last_message.content:
        content = last_message.content.lower()
        ending_keywords = [
            "goodbye", "see you", "no more questions", "ending", "stop",
            "bye", "have a nice day", "that's all", "nothing else",
            "no further tasks", "we are done"
        ]
        if any(keyword in content for keyword in ending_keywords):
            return 'exit'

    return 'call_model'


def should_continue(state: StateTravelTalk) -> Literal['call_model', "exit"]:
    """ Either End the conversation or continue talking """
    if is_ending_conversation(state["messages"]) == 'exit':
        return "exit"
    return "call_model"  # This should return the correct node name that exists, in this case "conversation"


def exit_node(state: StateTravelTalk, config: RunnableConfig):
    response = "Goodbye! Have a nice day."
    return {"messages": state["messages"] + [AIMessage(content=response)]}


workflow = StateGraph(StateTravelTalk)
workflow.add_node("call_model", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_node("exit", exit_node)

workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", "summarize_conversation")
workflow.add_conditional_edges('summarize_conversation', should_continue)
workflow.add_edge( "exit",END)




memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# ✅ Draw the graph
img_bytes = graph.get_graph().draw_mermaid_png()
img = mpimg.imread(BytesIO(img_bytes), format='png')
plt.imshow(img)
plt.axis('off')
plt.show()

print("GRAPH READY")



config = {"configurable": {"thread_id": "1"}}

messages = [HumanMessage(content="Want to go to italy and end the talk")]
messages = graph.invoke({"messages": messages}, config)

while True:
    for m in messages['messages']:
        m.pretty_print()
    
    # Ask the user for input
    user_input = input("Your input: ")  # Modify this line if you have a different input method

    # Check if the user input matches any ending conditions (e.g., "goodbye")
    if any(ending_word in user_input.lower() for ending_word in [
        "goodbye", "see you", "no more questions", "ending", "stop", "bye", "have a nice day", "that's all", "nothing else", "no further tasks", "we are done"
    ]):
        print("Ending the conversation.")
        break

    # Add the user input as a new HumanMessage
    messages['messages'].append(HumanMessage(content=user_input))

    # Invoke the graph to process the updated conversation
    messages = graph.invoke({"messages": messages['messages']}, config)

