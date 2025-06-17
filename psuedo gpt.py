"""import streamlit as st
import openai 

st.title("Pseudo ChatGPT")

openai.api_key = st.secrets["openai"]["api_key"]
#openai.api_key = st.secrets["openai"]["api_key"]
#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙇‍♂️"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})"""


import streamlit as st
import openai
import os

# Load OpenAI API key securely
def load_openai_key():
    if "openai" in st.secrets:
        openai.api_key = st.secrets["openai"]["api_key"]
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            openai.api_key = os.getenv("OPENAI_API_KEY")
        except ImportError:
            st.warning("Install python-dotenv if using a .env file locally.")

    if not openai.api_key:
        st.error("OpenAI API key not found.")
        st.stop()

# Ask OpenAI a question
def ask_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="OpenAI Q&A", layout="centered")
st.title("Ask OpenAI")

load_openai_key()

query = st.text_input("Ask something:")
if query:
    answer = ask_openai(query)
    st.write(answer)

