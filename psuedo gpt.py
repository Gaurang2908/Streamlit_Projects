# VERSION WITH DATA-AWARE GPT CONTEXT INJECTION

import streamlit as st
import pandas as pd
import joblib
import time
import tiktoken
from openai import OpenAI, RateLimitError, APIError

# Constants
MODEL = "gpt-3.5-turbo"
TPM_LIMIT = 90000

# Load OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load trained ML model
@st.cache_resource
def load_model():
    return joblib.load("readmission_model.pkl")

model = load_model()

# Load column definitions from text file
@st.cache_data
def load_column_definitions():
    try:
        with open("column_definitions.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "[Column definitions file not found.]"

column_reference = load_column_definitions()

# Token counting utility
def count_tokens(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for msg in messages:
        num_tokens += 4
        for key, value in msg.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2
    return num_tokens

# Scope filter
def is_bad_query(user_input: str) -> bool:
    text = user_input.lower().strip()
    blocklist = [
        "code", "python", "html", "celebrity", "movie", "joke", "riddle",
        "hack", "religion", "nude", "anime", "music"
    ]
    return any(term in text for term in blocklist)

# Summarize CSV for model context
def summarize_dataframe(df, max_rows=5):
    try:
        summary = df.describe(include='all').fillna('').astype(str)
        sample = df.head(max_rows).astype(str)
        return f"""
Dataset Summary:
Columns: {', '.join(df.columns)}
Describe:\n{summary.to_string()}
Sample rows:\n{sample.to_string()}
"""
    except Exception as e:
        return f"[Failed to summarize data: {e}]"

# Streamlit UI
st.title("Pseudo GPT")

uploaded_file = st.file_uploader("Upload healthcare data (CSV)", type="csv")
df = None
csv_uploaded = False

data_context = ""
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    csv_uploaded = True
    st.success("CSV file loaded successfully.")
    data_context = summarize_dataframe(df)

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask your healthcare-related question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if is_bad_query(prompt):
            response = "I'm designed to assist only with healthcare insights. Please rephrase."
        elif not csv_uploaded:
            response = "Please upload a healthcare CSV file first."
        else:
            context = [{"role": "system", "content": f"""
You are a healthcare analytics assistant. Use this real dataset summary and column definitions to answer questions.

--- Column Definitions ---
{column_reference}

--- Data Summary ---
{data_context}
"""}]
            context += st.session_state.messages[-10:]

            tokens_used = count_tokens(context)
            if tokens_used > TPM_LIMIT:
                time.sleep(60)

            max_retries = 3
            wait = 10
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model=MODEL,
                        messages=context,
                        stream=False,
                    )
                    response = completion.choices[0].message.content
                    break
                except RateLimitError:
                    time.sleep(wait)
                    wait *= 2
                except APIError as e:
                    response = f"OpenAI API error: {e}"
                    break
            else:
                response = "Rate limit error. Try again later."

        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
