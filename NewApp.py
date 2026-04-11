import time
import streamlit as st
from langchain_core.messages import SystemMessage

from MyStreamlitConfig import CSS_STYLE
from src import Streamlit_Agent

# --- 1. PAGE CONFIGURATION ---
# Note: st.set_page_config must be the very first Streamlit command called.
st.set_page_config(page_title="HR AI Agent | Hospital Portal", page_icon="🏥", layout="wide")

# --- 2. THEME & BRANDING ---
# Banner Image (Replace URL with your hospital's banner logo)
BANNER_URL = "imgs/CityHospital.png"
#st.image(BANNER_URL, width='stretch')

# Theme Toggle in Sidebar
with st.sidebar:
    #st.title("Settings")
    #theme_mode = st.toggle("🌙 Dark Mode", value=True)

    #st.divider()

    # --- 3. THE 25% PARTITION (Special Notes) ---
    st.subheader("📌 Supporting Information")
    st.info("""
    **System Status:** Online
    **Model:** HR-Specialist-v1

    *Notes:*
    - This agent has access to employee Payroll, LMS and Policies data.
    - You can apply leave from the agent by asking about leave balance or applying for leave.
    - Response times may vary based on query complexity.
    """)

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Apply Custom CSS from your config
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# Optional: Simple Dynamic Theme Injection
# if theme_mode:
#     st.markdown("""<style>div[data-testid="stToolbar"] {visibility: hidden;} </style>""", unsafe_allow_html=True)

# --- 4. INITIALIZE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 5. MAIN CHAT INTERFACE ---
st.title(":blue[HR AI Assistant for Hospital Staff]")
st.caption("Assisting hospital staff with HR and Policy inquiries.")

# Display existing messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. CHAT INPUT & LOGIC ---
if prompt := st.chat_input("Ask me anything about hospital policy..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing hospital database..."):
            try:
                # Calling your imported agent logic
                response, updated_history = Streamlit_Agent.run_repl(prompt, st.session_state.chat_history)
                st.session_state.chat_history = updated_history  # Update session history
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")