from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from tools import search_tool, calculator

load_dotenv()

# initializing database
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

# Initializing LLM
llm = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

# tool infused llm
tools = [search_tool, calculator]
llm_tools = llm.bind_tools(tools)

# LangGraph State
class ChatState(TypedDict):
   messages: Annotated[list[BaseMessage], add_messages]

# Node Function(s)
def chat_node(state: ChatState) -> dict:
    messages = state["messages"]
    response = llm_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# create graph
builder = StateGraph(ChatState)

# adding node
builder.add_node("chat_node", chat_node)
builder.add_node("tools", tool_node)

# creating edges
builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")

# compiling the graph
workflow = builder.compile(checkpointer=checkpoint)


def get_all_threads():
    all_threads = set()

    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)