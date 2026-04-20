import time
import streamlit as st
from langchain_core.messages import SystemMessage

from MyStreamlitConfig import CSS_STYLE
from src import Streamlit_Agent
from src.Streamlit_Agent import employee_type

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="HR AI Agent | CITY Hospital", page_icon="🏥", layout="wide")

# Apply Custom CSS
st.markdown(CSS_STYLE, unsafe_allow_html=True)

employee_type=None
# --- 2. AUTHENTICATION LOGIC ---
def login_page():
    """Displays a simple login form."""
    st.markdown("<h1 style='text-align: center;'>🏥CITY Hospital Staff Portal</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                # Replace these with your actual validation logic or database check
                if username == "user001" and password == "password":
                    st.session_state.user_role = "employee"
                    st.session_state.logged_in = True
                    st.success("Logged in successfully!")
                    st.rerun()
                elif username == "manager001" and password == "password":
                    st.session_state.user_role = "manager"
                    st.session_state.logged_in = True
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                     st.error("Invalid Username or Password")

def logout():
    """Clears the session state and returns to login."""
    st.session_state.logged_in = False
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()


# --- 3. SESSION STATE INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None  # Initialize role
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 4. CONDITIONAL RENDERING ---
if not st.session_state.logged_in:
    login_page()
else:
    # --- SIDEBAR (ONLY VISIBLE AFTER LOGIN) ---
    with st.sidebar:
        st.subheader("📌 Supporting Information")
        st.info("""
        **System Status:** Online
        **Model:** HR-Specialist-v1

        *Notes:*
        - Access to Payroll, LMS and Policies.
        """)

        st.divider()

        # Logout Action
        if st.button("Logout", use_container_width=True):
            logout()

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- MAIN CHAT INTERFACE ---
    st.title(":blue[CITY Hospital - HR AI Assistant ]")
    st.caption("Assisting hospital staff with HR and Policy inquiries.")

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # CHAT INPUT & LOGIC
    if prompt := st.chat_input("Ask me anything about hospital policy..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing hospital database..."):
                try:
                    Streamlit_Agent.write_employee_type(st.session_state.user_role)
                    response, updated_history = Streamlit_Agent.run_repl(prompt, st.session_state.chat_history)
                    st.session_state.chat_history = updated_history
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")