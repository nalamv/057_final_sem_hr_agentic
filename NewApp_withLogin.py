import warnings
warnings.filterwarnings("ignore", module="transformers")

import time
import os
import pandas as pd
from datetime import datetime

import streamlit as st
from langchain_core.messages import SystemMessage

from MyStreamlitConfig import CSS_STYLE

#from src import Streamlit_Agent
from src.Streamlit_Agent import employee_type, employee_id

# ← CHANGED: Use LangGraph version
from src import agent_langgraph as Streamlit_Agent
from src.agent_langgraph import employee_type


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="HR AI Agent | VNIT Health Care Nagpur",
    page_icon="🏥",
    layout="wide"
)

# Apply Custom CSS
st.markdown(CSS_STYLE, unsafe_allow_html=True)


# =========================================================
# FEEDBACK LOGGING FUNCTION
# =========================================================
FEEDBACK_FILE = "feedback_logs.csv"


def save_feedback(employee_id, role, question, answer, feedback_type):
    """
    Save feedback into CSV log file.
    """

    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "employee_id": employee_id,
        "role": role,
        "question": question,
        "answer": answer,
        "feedback": feedback_type
    }

    df = pd.DataFrame([feedback_data])

    file_exists = os.path.isfile(FEEDBACK_FILE)

    df.to_csv(
        FEEDBACK_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )


# =========================================================
# LOGIN PAGE
# =========================================================
def login_page():
    """Displays a simple login form."""

    st.markdown(
        "<h1 style='text-align: center;'>🏥VNIT Health Care Employee Portal</h1>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):

            st.subheader("Login")

            username = st.text_input("Username")

            password = st.text_input("Password", type="password")

            submit = st.form_submit_button("Login")

            if submit:

                # Employee Login
                if username.startswith("emp"):

                    employee_id = username[3:]

                    st.session_state.user_role = "employee"

                    st.session_state.employee_id = employee_id

                    st.session_state.logged_in = True

                    st.success(
                        f"Logged in successfully as Employee {employee_id}!"
                    )

                    st.rerun()

                # Manager Login
                elif username.startswith("super"):

                    employee_id = username[5:]

                    st.session_state.user_role = "manager"

                    st.session_state.employee_id = employee_id

                    st.session_state.logged_in = True

                    st.success(
                        f"Logged in successfully as Manager {employee_id}!"
                    )

                    st.rerun()

                else:

                    st.error(
                        "Invalid Username or Password. "
                        "Use 'emp<id>' for employees "
                        "or 'super<id>' for managers."
                    )


# =========================================================
# LOGOUT FUNCTION
# =========================================================
def logout():
    """
    Clears the session state and returns to login.
    """

    st.session_state.logged_in = False
    st.session_state.messages = []
    st.session_state.chat_history = []

    st.rerun()


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "employee_id" not in st.session_state:
    st.session_state.employee_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# CONDITIONAL RENDERING
# =========================================================
if not st.session_state.logged_in:

    login_page()

else:

    # =====================================================
    # SIDEBAR
    # =====================================================
    with st.sidebar:

        st.subheader("📌 Supporting Information")

        st.info(f"""
        **System Status:** Online \n
        **Role:** :green[{st.session_state.user_role}] \n
        **Employee ID:** :blue[{st.session_state.employee_id}]

        *Notes:*
        - Access to Payroll, LMS and Policies.
        """)

        st.divider()

        # Logout
        if st.button("Logout", use_container_width=True):
            logout()

        # Clear Chat
        if st.button("Clear Chat History", use_container_width=True):

            st.session_state.messages = []

            st.session_state.chat_history = []

            st.rerun()

    # =====================================================
    # MAIN CHAT INTERFACE
    # =====================================================
    st.title(":blue[VNIT Health Care Employee Portal (AI Assistant)]")

    st.caption(
        "Assisting hospital staff with HR services, "
        "leave management, payroll support, hospital policies, "
        "LMS access, and employee-related inquiries"
    )

    # =====================================================
    # DISPLAY EXISTING MESSAGES
    # =====================================================
    for idx, message in enumerate(st.session_state.messages):

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            # =================================================
            # FEEDBACK BUTTONS FOR ASSISTANT RESPONSES
            # =================================================
            if message["role"] == "assistant":

                feedback_given = message.get("feedback", None)

                if feedback_given is None:

                    col1, col2, col3 = st.columns([1, 1, 8])

                    # 👍 THUMBS UP
                    with col1:

                        if st.button("👍", key=f"up_{idx}"):

                            st.session_state.messages[idx]["feedback"] = "positive"

                            # Find previous user question
                            question = ""

                            if idx > 0:
                                question = st.session_state.messages[idx - 1]["content"]

                            # Save feedback
                            save_feedback(
                                employee_id=st.session_state.employee_id,
                                role=st.session_state.user_role,
                                question=question,
                                answer=message["content"],
                                feedback_type="positive"
                            )

                            st.success("Positive feedback recorded!")

                            st.rerun()

                    # 👎 THUMBS DOWN
                    with col2:

                        if st.button("👎", key=f"down_{idx}"):

                            st.session_state.messages[idx]["feedback"] = "negative"

                            # Find previous user question
                            question = ""

                            if idx > 0:
                                question = st.session_state.messages[idx - 1]["content"]

                            # Save feedback
                            save_feedback(
                                employee_id=st.session_state.employee_id,
                                role=st.session_state.user_role,
                                question=question,
                                answer=message["content"],
                                feedback_type="negative"
                            )

                            st.warning("Negative feedback recorded!")

                            st.rerun()

                else:

                    # Already Submitted
                    if feedback_given == "positive":
                        st.success("👍 Feedback Submitted")

                    elif feedback_given == "negative":
                        st.warning("👎 Feedback Submitted")

    # =====================================================
    # CHAT INPUT
    # =====================================================
    if prompt := st.chat_input("Ask me anything about hospital policy..."):

        # =================================================
        # USER MESSAGE
        # =================================================
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # =================================================
        # ASSISTANT RESPONSE
        # =================================================
        with st.chat_message("assistant"):

            with st.spinner("Analyzing hospital database..."):

                try:

                    Streamlit_Agent.write_employee_type(
                        st.session_state.user_role
                    )

                    Streamlit_Agent.write_employee_id(
                        st.session_state.employee_id
                    )

                    response, updated_history = Streamlit_Agent.run_repl(
                        prompt,
                        st.session_state.chat_history,
                        st.session_state.employee_id,
                        st.session_state.user_role
                    )

                    # Update history
                    st.session_state.chat_history = updated_history

                    # Display response
                    st.markdown(response)

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "feedback": None
                    })

                    st.rerun()

                except Exception as e:

                    st.error(f"An error occurred: {e}")