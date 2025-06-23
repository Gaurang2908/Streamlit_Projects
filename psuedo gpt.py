import streamlit as st
from openai import OpenAI, RateLimitError, APIError
import time
import tiktoken
import pandas as pd  # ✅ [ADDED] for CSV handling

# Constants
MODEL = "gpt-3.5-turbo"
TPM_LIMIT = 90000  # Adjust this if your quota is higher

# API client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ [ADDED] System prompt to define assistant scope
SYSTEM_PROMPT = """
You are a corporate wellness analytics assistant for HR and business leaders.

ONLY respond to questions about:
- Employee wellness
- Corporate health data
- Insights from health-related dashboards
- Absenteeism, presenteeism, productivity
- Workplace mental and physical health trends

If the user greets you or thanks you, respond politely and guide them back to wellness insights.

DO NOT answer questions unrelated to workplace wellness (e.g., code, philosophy, politics, general trivia). 
If asked such questions, politely say: "I'm designed to assist only with corporate health & wellness insights."

ALWAYS use the uploaded CSV data to answer questions. If information is missing, could you say so? Do not hallucinate.
I'd appreciate it if you could be concise, insightful, and focused on business impact.
"""

# ✅ [ADDED] Check if query is wellness-related
def is_bad_query(user_input: str) -> bool:
    # Allow casual/neutral things like greetings
    """greetings = ["hi", "hello", "hey", "thanks", "thank you"]
    if any(greet in user_input.lower() for greet in greetings):
        return False
    """

    # Block only clearly unrelated things
    blocklist = [
        "code", "python", "javascript", "capital of", "president", "game", "movie",
        "joke", "riddle", "love", "god", "religion", "death", "kill", "hack"
    ]
    return any(word in user_input.lower() for word in blocklist)

# ✅ [ADDED] File uploader for CSV data
uploaded_file = st.file_uploader("Upload your corporate wellness data (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_description = df.describe(include='all').to_string()
    df_columns = ", ".join(df.columns)
    data_context = f"""
    This is the dataset you must use to answer all queries.
    
    Available columns: {df_columns}
    
    Dataset Summary (stats, categories, etc):
    {df_description}
    """
    else:
        data_context = ""  # Safe fallback if no file is uploaded

# Token counter
def count_tokens(messages, model="gpt-3.5-turbo"):
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
    with st.chat_message("user", avatar="🙇‍♂️"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="👾"):

        # ✅ [ADDED] Filter query scope before processing
        if is_bad_query(prompt):
            response = "I'm designed to assist only with corporate health & wellness insights. Please rephrase your question."
        elif not uploaded_file:
            response = "Please upload a CSV file with corporate wellness data before asking questions."
        else:
            # ✅ [MODIFIED] Include system prompt + dataset context
            context = [
                {"role": "system", "content": SYSTEM_PROMPT} + st.session_state.messages[-10],
                {"role": "user", "content": f"{data_context}\n\nUser Query: {prompt}"}
            ]

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
                    st.warning(f"⏳ Rate limit hit. Retrying in {wait} sec... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    wait *= 2

                except APIError as e:
                    st.error(f"💥 OpenAI API error: {e}")
                    response = "[API Error]"
                    break

            if not success:
                response = "⚠️ Failed after multiple retries due to rate limits. Please try again later."

    st.session_state.messages.append({"role": "assistant", "content": response})
