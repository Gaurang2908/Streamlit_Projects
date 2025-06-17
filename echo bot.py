import streamlit as st
import numpy as np

with st.chat_message("assistant", avatar="🤖"):
    st.write("Hello human")
    st.line_chart(np.random.randn(40, 4))

"""prompt = st.chat_input("Write something")
if prompt :
    st.write(f"User has sent the following prompt: {prompt}")"""

st.title("Echo Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    with st.chat_message("user", avatar="🙇‍♂️"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user", "content": prompt})

response = f"Echo: {prompt}"
with st.chat_message("assistant", avatar="🤖"):
    st.markdown(response)
st.session_state.messages.append({"role": "assistant", "content": response})
