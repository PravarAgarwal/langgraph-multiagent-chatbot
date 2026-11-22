import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from chatbot_backend import get_all_threads, workflow


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = workflow.get_state(config)
        if state.values and "messages" in state.values:
            return state.values["messages"]
    except Exception:
        pass
    return []


def create_new_chat() -> None:
    new_id = generate_thread_id()
    st.session_state["current_thread_id"] = new_id
    add_thread(new_id)
    st.session_state["messages"] = []


def extract_text_content(content) -> str:
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def format_tool_label(tool_name: str) -> str:
    friendly_names = {
        "search_tool": "Searching the web",
        "calculator": "Running calculator",
    }
    return friendly_names.get(tool_name, f"Using `{tool_name}`")


def build_tool_results_map(messages: list) -> dict[str, ToolMessage]:
    results_by_call_id: dict[str, ToolMessage] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            call_id = getattr(msg, "tool_call_id", None) or getattr(msg, "id", None)
            if call_id:
                results_by_call_id[call_id] = msg
    return results_by_call_id


def render_past_tool_calls(
    tool_calls: list[dict], results_by_call_id: dict[str, ToolMessage]
) -> None:
    if not tool_calls:
        return

    tool_names = ", ".join(call.get("name", "tool") for call in tool_calls)
    with st.expander(f"🔧 Tools used ({tool_names})", expanded=False):
        for idx, call in enumerate(tool_calls, start=1):
            call_name = call.get("name", "tool")
            call_args = call.get("args", {})
            call_id = call.get("id")
            tool_msg = results_by_call_id.get(call_id) if call_id else None

            st.markdown(f"**{idx}. `{call_name}`**")
            if call_args:
                st.json(call_args)
            if tool_msg is not None:
                st.caption("Output")
                st.write(tool_msg.content)


st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered",
)

# Hydrate persisted thread IDs from SQLite once per Streamlit session
if "chat_threads" not in st.session_state:
    persisted_threads = get_all_threads()
    st.session_state["chat_threads"] = list(persisted_threads)

if "current_thread_id" not in st.session_state:
    create_new_chat()

if "messages" not in st.session_state:
    st.session_state["messages"] = load_conversation(st.session_state["current_thread_id"])

# --- Sidebar UI ---
with st.sidebar:
    st.header("💬 Chat Threads")

    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.divider()
    st.subheader("Your Conversations")

    for thread_id in reversed(st.session_state["chat_threads"]):
        is_active = thread_id == st.session_state["current_thread_id"]
        button_type = "primary" if is_active else "secondary"
        label = f"Thread {thread_id[:8]}..."

        if st.button(label, key=thread_id, type=button_type, use_container_width=True):
            st.session_state["current_thread_id"] = thread_id
            st.session_state["messages"] = load_conversation(thread_id)
            st.rerun()

# --- Main Chat UI ---
st.title("🤖 LangGraph Chatbot")
st.write("Powered by Google Gemini and LangGraph Memory.")

config = {
    "configurable": {"thread_id": st.session_state["current_thread_id"]}
}

# Replay persisted conversation history with collapsed past tool summaries
tool_results_map = build_tool_results_map(st.session_state["messages"])

for message in st.session_state["messages"]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        tool_calls = getattr(message, "tool_calls", None) or []
        text_content = extract_text_content(message.content).strip()

        if tool_calls or text_content:
            with st.chat_message("assistant"):
                if tool_calls:
                    render_past_tool_calls(tool_calls, tool_results_map)
                if text_content:
                    st.markdown(text_content)
    elif isinstance(message, ToolMessage):
        continue

if user_input := st.chat_input("What would you like to chat about?"):
    add_thread(st.session_state["current_thread_id"])

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_placeholder = st.container()
        stream_state = {"active_status": None}

        def generate_response_stream():
            for msg_chunk, metadata in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                node_name = (metadata or {}).get("langgraph_node", "")

                # Tool execution output coming back from the `tools` node
                if isinstance(msg_chunk, ToolMessage) or node_name == "tools":
                    tool_name = getattr(msg_chunk, "name", None) or "tool"
                    if stream_state["active_status"] is None:
                        with status_placeholder:
                            stream_state["active_status"] = st.status(
                                f"{format_tool_label(tool_name)}...",
                                expanded=False,
                            )
                    stream_state["active_status"].update(
                        label=f"Finished {format_tool_label(tool_name).lower()}",
                        state="complete",
                    )
                    with stream_state["active_status"]:
                        st.caption(f"Tool: `{tool_name}`")
                        st.write(msg_chunk.content)
                    continue

                # AI message chunk: may contain tool_calls and/or text tokens
                if isinstance(msg_chunk, AIMessage) or getattr(msg_chunk, "type", "") == "ai":
                    tool_calls = getattr(msg_chunk, "tool_calls", None) or []
                    if tool_calls:
                        for call in tool_calls:
                            call_name = call.get("name", "tool")
                            call_args = call.get("args", {})
                            with status_placeholder:
                                stream_state["active_status"] = st.status(
                                    f"{format_tool_label(call_name)}...",
                                    expanded=True,
                                )
                                st.write(f"**Tool:** `{call_name}`")
                                if call_args:
                                    st.json(call_args)

                    chunk_content = msg_chunk.content
                    if isinstance(chunk_content, list):
                        for item in chunk_content:
                            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                                yield item["text"]
                            elif isinstance(item, dict) and "text" in item and item["text"]:
                                yield item["text"]
                            elif isinstance(item, str) and item:
                                yield item
                    elif isinstance(chunk_content, str) and chunk_content:
                        yield chunk_content

            if stream_state["active_status"] is not None:
                stream_state["active_status"].update(state="complete", expanded=False)

        st.write_stream(generate_response_stream())

    updated_state = workflow.get_state(config)
    if updated_state.values and "messages" in updated_state.values:
        st.session_state["messages"] = updated_state.values["messages"]