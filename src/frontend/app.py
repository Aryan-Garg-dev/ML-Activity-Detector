import streamlit as st
import requests
import json
import uuid

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AML Activity Detector Agent",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


def ingest_datasets(accounts_file, transactions_file, alerts_file):
    files = {}
    if accounts_file:
        files["accounts"] = (accounts_file.name, accounts_file.getvalue(), "text/csv")
    if transactions_file:
        files["transactions"] = (transactions_file.name, transactions_file.getvalue(), "text/csv")
    if alerts_file:
        files["alerts"] = (alerts_file.name, alerts_file.getvalue(), "text/csv")

    if not files:
        st.warning("Please select at least one dataset to upload.")
        return

    with st.spinner("Uploading and ingesting datasets..."):
        try:
            response = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=60)
            if response.status_code == 200:
                data = response.json()
                st.success("Datasets ingested successfully!")
                st.json(data["ingested_rows"])
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"Failed to ingest datasets: {detail}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API server at `http://localhost:8000`. Is uvicorn running?")
        except requests.exceptions.Timeout:
            st.error("Ingest request timed out (60s). The dataset may be too large.")
        except Exception as e:
            st.error(f"Error connecting to the API: {str(e)}")


def send_query_stream(query, provider, model_id, detailed):
    payload = {
        "query": query,
        "thread_id": st.session_state.thread_id
    }
    if provider:
        payload["provider"] = provider
    if model_id:
        payload["model_id"] = model_id

    params = {"detailed": detailed}

    try:
        response = requests.post(
            f"{API_BASE_URL}/stream",
            json=payload,
            params=params,
            stream=True,
            timeout=(10, 300),  # 10s connect, 300s read (agent can take time)
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            yield {"event": "error", "detail": f"API returned {response.status_code}: {detail}"}
            return

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        pass  # Skip malformed lines silently

    except requests.exceptions.ConnectionError:
        yield {"event": "error", "detail": "Cannot connect to API server. Is uvicorn running on port 8000?"}
    except requests.exceptions.Timeout:
        yield {"event": "error", "detail": "Request timed out waiting for agent response (300s). Try a simpler query."}
    except Exception as e:
        yield {"event": "error", "detail": f"Connection error: {str(e)}"}


# Sidebar for settings and dataset upload
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox("LLM Provider (optional)", options=["", "groq", "lmstudio", "openrouter"], index=0)
    model_id = st.text_input("Model ID (optional)", placeholder="e.g. llama-3.3-70b-versatile")
    detailed = st.checkbox("Request detailed response", value=False)

    st.divider()

    # API health check
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health.status_code == 200:
            h = health.json()
            status_color = "🟢" if h.get("status") == "healthy" else "🟡"
            st.caption(f"{status_color} API: {h.get('status', 'unknown')} | DB: {'✅' if h.get('duckdb_connected') else '❌'} | Provider: `{h.get('active_provider', 'N/A')}`")
        else:
            st.caption("🔴 API: unreachable")
    except Exception:
        st.caption("🔴 API: not running")

    st.divider()

    st.header("📤 Dataset Upload")
    st.caption("Upload CSV datasets to override existing tables.")
    accounts_file = st.file_uploader("Accounts CSV", type=["csv"])
    transactions_file = st.file_uploader("Transactions CSV", type=["csv"])
    alerts_file = st.file_uploader("Alerts CSV", type=["csv"])

    if st.button("Ingest Datasets", type="primary"):
        ingest_datasets(accounts_file, transactions_file, alerts_file)

    st.divider()
    st.caption(f"Session Thread ID: `{st.session_state.thread_id}`")
    if st.button("Reset Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# Main Application Area
st.title("🕵️‍♂️ AML Suspicious Activity Agent")
st.markdown("""
Welcome to the Autonomous AML Agent Platform. Upload your CSV datasets on the left, then ask natural language queries to investigate suspicious activity.
""")

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "data" in message:
            with st.expander("Agent Response Details", expanded=False):
                st.json(message["data"])

# Chat input
if query := st.chat_input("E.g., Find structuring patterns in the last 30 days"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process agent response
    with st.chat_message("assistant"):
        status_placeholder = st.status("🧠 Agent analyzing query...", expanded=True)

        agent_data = None
        error_occurred = False

        with status_placeholder:
            st.write("Initializing tool graph & semantic cache lookup...")

            for event in send_query_stream(query, provider if provider else None, model_id if model_id else None, detailed):
                event_type = event.get("event")

                if event_type == "node_update":
                    node_name = event.get("node", "unknown")
                    # Map internal node names to human-readable labels
                    node_labels = {
                        "parse_intent": "🔍 Parsing query intent",
                        "tool_agent": "🤖 Agent reasoning",
                        "tools": "🛠️ Executing tools",
                        "extract_state": "📊 Extracting results",
                        "fallback_fixed_flow": "🔄 Running fallback pipeline",
                        "clarify": "❓ Clarifying query",
                        "explain": "📝 Generating explanation",
                        "verify": "✅ Verifying results",
                        "report": "📋 Assembling report",
                    }
                    label = node_labels.get(node_name, f"⚙️ {node_name}")
                    st.write(f"- {label}")

                elif event_type == "complete":
                    agent_data = event.get("data")
                    status_placeholder.update(label="✅ Query execution complete", state="complete", expanded=False)

                elif event_type == "error":
                    error_occurred = True
                    err_detail = event.get("detail", "Unknown error")
                    # Show the real error message — never "UNKNOWN"
                    err_msg = err_detail if err_detail.strip() else f"Unexpected error (type: {event.get('type', 'unknown')})"
                    st.error(f"**Agent error:** {err_msg}")
                    # Optionally show traceback in expander for debugging
                    if event.get("traceback"):
                        with st.expander("🔍 Error Details (traceback)", expanded=False):
                            st.code(event["traceback"], language="python")
                    status_placeholder.update(label="❌ Query failed", state="error", expanded=False)
                    break

        if agent_data:
            # Display explanation
            explanation = agent_data.get("explanation", "No explanation provided.")
            st.markdown(explanation)

            # Show flagged items in an expander if they exist
            flagged_items = agent_data.get("flagged_items", [])
            if flagged_items:
                with st.expander(f"🚩 Flagged Items ({len(flagged_items)})", expanded=True):
                    st.dataframe(flagged_items)

            # Show execution summary in an expander
            summary = agent_data.get("execution_summary", {})
            if summary:
                with st.expander("⚙️ Execution Summary"):
                    st.json(summary)

            # Save assistant message to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": explanation,
                "data": agent_data
            })
        elif not error_occurred:
            # Agent ran but produced no response
            st.warning("Agent completed but produced no response. Check the server logs.")
