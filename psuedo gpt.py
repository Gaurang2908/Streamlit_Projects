import streamlit as st
import openai 
from openai import OpenAI

st.title("Pseudo ChatGPT")

#openai.api_key = st.secrets["api_key"]
#openai.api_key = st.secrets["openai"]["api_key"]
#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            # Limit token history to avoid overload (optional but useful)
            context = st.session_state.messages[-10:]

            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=context,
                stream=True,
            )
            response = st.write_stream(stream)

        except RateLimitError:
            st.warning(" OpenAI Rate limit hit. Waiting 5 seconds...")
            time.sleep(5)
            try:
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=context,
                    stream=True,
                )
                response = st.write_stream(stream)
            except RateLimitError:
                st.error("❌ Still rate-limited. Try again in a minute.")
                response = "[Rate limited]"

        except APIError as e:
            st.error(f"⚠️ OpenAI API Error: {e}")
            response = "[API Error]"
    # Save assistant reply to state
    st.session_state.messages.append({"role": "assistant", "content": response})
