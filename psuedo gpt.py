import streamlit as st
from openai import OpenAI, RateLimitError, APIError
import time
import tiktoken

# Constants
MODEL = "gpt-4"
TPM_LIMIT = 100000  # Adjust based on your plan and model quota

# API Client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Token counter
def count_tokens(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for msg in messages:
        num_tokens += 4  # every message metadata overhead
        for key, value in msg.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # priming tokens
    return num_tokens

# UI
st.title("PseudoGPT")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context = st.session_state.messages[-10:]
        tokens_used = count_tokens(context)

        if tokens_used > TPM_LIMIT:
            st.warning(f"⚠️ Token usage ({tokens_used}) exceeds limit ({TPM_LIMIT}). Waiting 60s.")
            time.sleep(60)

        max_retries = 3
        wait = 10
        success = False

        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=context,
                    stream=False,
                )
                response = completion.choices[0].message.content
                st.markdown(response)
                success = True
                break

            except RateLimitError:
                st.warning(f"⏳ Rate limit hit. Retrying in {wait} sec... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                wait *= 2

            except APIError as e:
                st.error(f"💥 OpenAI API error: {e}")
                response = "[API Error]"
                break

        if not success:
            response = "⚠️ Failed after multiple retries due to rate limits. Please try again later."

    st.session_state.messages.append({"role": "assistant", "content": response})

    st.session_state.messages.append({"role": "assistant", "content": response})
