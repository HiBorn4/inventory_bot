import streamlit as st
import requests
import os
import json

# Configuration
def get_api_url():
    return os.getenv("API_URL", "http://localhost:8000")

API_URL = get_api_url()

st.set_page_config(page_title="Application Inventory Chatbot", layout="centered")
st.title("📱 Application Inventory Chatbot")
st.write("Ask questions about your application inventory and get instant answers.")

# Input form
def user_query_form():
    with st.form(key="query_form", clear_on_submit=True):
        question = st.text_input(
            "Your question:",
            "",
            placeholder="e.g. Which applications are exposed on the public internet?"
        )
        submit = st.form_submit_button(label="Ask")
    return question, submit

question, submitted = user_query_form()

if submitted:
    if not question.strip():
        st.error("Please enter a question before submitting.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()

                # Extract and display fields cleanly
                query_text = data.get("query", question)
                result_obj = data.get("result", data.get("answer", {}))

                if isinstance(result_obj, str):
                    # Try to load JSON if result is JSON string
                    try:
                        result_obj = json.loads(result_obj)
                    except json.JSONDecodeError:
                        result_obj = {"response": result_obj}

                st.subheader("Your Question")
                st.write(query_text)

                st.subheader("Answer")
                st.write(result_obj.get("result", "No result found."))


                with st.expander("Show raw response"):
                    st.json(data)

            except requests.exceptions.RequestException as e:
                st.error(f"Error contacting the backend: {e}")
