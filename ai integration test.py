import streamlit as st
import numpy as np

with st.chat_message("assistant"):
    st.write("Hello human")
    st.bar_chart(np.random.randn(30, 3))

prompt = st.chat_input("Type here")
if prompt :
    st.write(f"User has sent the following prompt:{prompt}")
