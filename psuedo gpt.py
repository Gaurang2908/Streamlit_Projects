import streamlit as st
from openai import OpenAI, RateLimitError, APIError
import pandas as pd
import time
import tiktoken

# Constants
MODEL = "gpt-3.5-turbo"
TPM_LIMIT = 90000

# OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Define assistant scope
SYSTEM_PROMPT = """
You are a corporate wellness analytics assistant for HR and business leaders.

ONLY respond to questions about:
- Employee wellness
- Corporate health data
- Insights from health-related dashboards
- Absenteeism, presenteeism, productivity
- Workplace mental and physical health trends

DO NOT answer questions unrelated to workplace wellness (e.g., code, philosophy, politics, general trivia). 
If asked such questions, politely say: "I'm designed to assist only with corporate health & wellness insights."

If greeted or thanked, respond politely and guide users back to wellness insights.

Always use the uploaded CSV data to answer questions. If information is missing, say so. Do not hallucinate.
"""

# Smart query filter
def is_bad_query(user_input: str) -> bool:
    text = user_input.lower().strip()

    # Block only obviously unrelated/offensive queries
    blocklist = [
        "code", "python", "javascript", "html", "css", "capital of", "president",
        "celebrity", "movie", "joke", "riddle", "religion", "god", "hack", "kill", 
        "murder", "bomb", "sex", "nude", "dating", "love", "game", "anime", "song", "music"
    ]

    return any(term in text for term in blocklist)

# Token counter
def count_tokens(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for msg in messages:
        num_tokens += 4
        for key, value in msg.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2
    return num_tokens

# UI
st.title("PseudoGPT")

uploaded_file = st.file_uploader("Upload your corporate wellness data (CSV)", type="csv")

df = None
data_context = ""
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df_description = df.describe(include='all').to_string()
        df_columns = ", ".join(df.columns)
        data_context = f"""
This is the dataset you must use to answer all queries.

Available columns: {df_columns}

Dataset Summary:
{df_description}
"""
    except Exception as e:
        st.error(f"❌ Failed to read the CSV file: {e}")
        data_context = "⚠️ CSV could not be read. No context available."

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙇‍♂️"):
        st.markdown(prompt)

    # Process assistant reply logic
    if is_bad_query(prompt):
        response = "🚫 I'm designed to assist only with corporate health & wellness insights. Please rephrase your question."
    elif df is None:
        response = "📂 Please upload a valid CSV file to get insights."
    else:
        # GPT logic
        context = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages[-10:]
        context[1]["content"] = f"{data_context}\n\nUser Query: {prompt}"

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

    # AFTER GPT call finishes...
    with st.chat_message("assistant", avatar="👾"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

