import streamlit as st
import numpy as np

with st.chat_message("assistant"):
    st.write("Hello human")
    st.line_chart(np.random.randn(40, 4))

prompt = st.chat_input("Write something")
if prompt :
    st.write(f"User has sent the following prompt: {prompt}")
