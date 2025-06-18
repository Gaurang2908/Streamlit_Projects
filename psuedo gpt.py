import streamlit as st
from openai import OpenAI, RateLimitError, APIError
import time

# Title
st.title("ChatGPT-like clone 🧠")

# API setup
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Model
MODEL = "gpt-3.5-turbo"

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context = st.session_state.messages[-10:]  # Limit context

        max_retries = 3
        wait = 5
        success = False

        for attempt in range(max_retries):
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=context,
                    stream=True,
                )
                response = st.write_stream(stream)
                success = True
                break

            except RateLimitError:
                st.warning(f"⏳ Rate limit hit. Retrying in {wait} sec... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                wait *= 2  # exponential backoff

            except APIError as e:
                st.error(f"💥 OpenAI API error: {e}")
                response = "[API Error]"
                break

        if not success:
            response = "⚠️ Failed after multiple retries due to rate limits. Please try again later."

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
