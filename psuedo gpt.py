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
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import openai
import os

# Load OpenAI key securely
def load_openai_key():
    if "openai" in st.secrets:
        openai.api_key = st.secrets["api_key"]
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            openai.api_key = os.getenv("OPENAI_API_KEY")
        except ImportError:
            st.warning("python-dotenv not found. Install it locally for .env support.")

    if not openai.api_key:
        st.stop()  # Prevents rest of app from running
        st.error("OpenAI API key not found. Please set it via Streamlit secrets or a .env file.")

# Use OpenAI to generate insight code (if needed)
def ask_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data assistant. Use pandas to answer queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None

# Load FAISS index and insights
def load_index_and_insights():
    with open("embeddings/faiss_index.pkl", "rb") as f:
        index, insights = pickle.load(f)
    return index, insights

# Load Sentence Transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Search for best matching insight
def search_insight(query, model, index, insights):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=1)
    return insights[I[0][0]]

# Streamlit UI
st.set_page_config(page_title="Analytics Chatbot", layout="centered")
st.title("Wellness Analytics Assistant")

query = st.text_input("Ask me a question about the dashboard:")

if query:
    load_openai_key()
    model = load_model()
    index, insights = load_index_and_insights()
    matched = search_insight(query, model, index, insights)

    st.subheader("Insight:")
    st.write(matched['insight'])

    if matched.get('chart_path'):
        try:
            img = Image.open(matched['chart_path'])
            st.image(img, caption="Chart", use_column_width=True)
        except FileNotFoundError:
            st.warning("Chart image not found.")

    # Optional: Ask OpenAI for pandas code (placeholder demo)
    if st.button("Run with AI"):
        schema = "Columns: anxiety_score, depression_score, location, date, nps"
        prompt = f"{query}\nSchema: {schema}\nReturn Python pandas code only."
        code = ask_openai(prompt)
        if code:
            st.code(code, language='python')

