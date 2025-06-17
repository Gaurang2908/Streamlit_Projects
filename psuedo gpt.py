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
import os
from openai import OpenAI

# Load OpenAI API key using migrate method
client = OpenAI(
    api_key=st.secrets["api_key"] 
    if "openai" in st.secrets else os.getenv("OPENAI_API_KEY")
)

if not client.api_key:
    st.error("OpenAI API key not found.")
    st.stop()

# Ask OpenAI a question using the migrate method and client.completions.create

def ask_openai(prompt):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.2
    )
    return response.choices[0].text.strip()

# Streamlit UI
st.set_page_config(page_title="OpenAI Q&A", layout="centered")
st.title("Ask OpenAI")

query = st.text_input("Ask something:")
if query:
    answer = ask_openai(query)
    st.write(answer)
