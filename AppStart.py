import time
import streamlit as st

from MyStreamlitConfig import CSS_STYLE
from src import Streamlit_Agent

# --- 1. SET UP THE PAGE ---
st.markdown(CSS_STYLE,unsafe_allow_html=True)
st.set_page_config(page_title="HR AI Agent", page_icon="🤖",layout="centered")
st.title("Welcome to HR AI Agent")

# --- 2. INITIALIZE CHAT HISTORY ---
# Streamlit reruns the whole script on every interaction.
# We use session_state to keep the messages persistent.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. DISPLAY EXISTING MESSAGES ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. THE AGENT LOGIC (Placeholder) ---
def call_my_agent(query):
    # This is where you call your LLM or Agent function
    # result = my_agent.invoke(query)
    time.sleep(1)  # Simulating processing time
    return f"The Agent processed your request: '{query}'"


# --- 5. CHAT INPUT & RESPONSE ---
if prompt := st.chat_input("Ask me anything..."):
    # Display user message in chat container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to session history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = Streamlit_Agent.run_repl(prompt)
            st.markdown(response)

    # Add assistant response to session history
    st.session_state.messages.append({"role": "assistant", "content": response})