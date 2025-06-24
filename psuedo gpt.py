# VERSION WITH AUTO-PREDICTION REMOVED AND COLUMN DEFINITIONS CONTEXT ADDED

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

# System prompt for healthcare assistant
SYSTEM_PROMPT = f"""
You are a healthcare analytics assistant. You help users interpret and analyze healthcare-related data from CSV files.

Here are the column definitions for reference:
Column definitions- 
Patient ID - Unique ID associated with each patient
Age - Age of the individual, ranging from 20 to 85
Gender - (Here)M/F
BMI - BMI is a tool that estimates the amount of body fat by using height and weight measurements. It can help assess health risks, but it has some limitations and does not diagnose conditions. Calculated using BMI = (Weight)/(Height in meters)^2
Smoker: Indicates if the patient is a smoker (Yes/No)
BMI: Body Mass Index of the patient
Diabetes: Indicates if the patient has diabetes (Yes/No)
Exercise Frequency (Days/week) - Number of days  a patient exercises in a week (out of 7) - ranges from 0 to 7
Cholesterol level - Cholesterol levels show how much cholesterol is circulating in your blood.(LDL + HDL) 100 to 200 (usual range for metric)
Hospital visit data (for last year) - Number of hospital visits by the patient in the last year 
Readmissions within 30 days - If the patient was readmitted to the hospital within 30 days of being discharged


You can assist with:
- Public and community health
- Clinical outcomes
- Utilization, claims, or EMR data
- Digital health trends and metrics
- Wellness, chronic conditions, mental health
- Epidemiological or demographic health trends
- Patient engagement and health program effectiveness

Always use the uploaded CSV before answering any question and to support your responses. If the data is insufficient or irrelevant, say so — don’t guess.
Avoid answering questions unrelated to healthcare (e.g., programming, politics, jokes, general trivia).
"""

# Streamlit UI
st.title("Pseudo gpt")

uploaded_file = st.file_uploader("Upload healthcare data (CSV)", type="csv")
df = None
csv_uploaded = False

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    csv_uploaded = True
    st.success("CSV file loaded successfully.")

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
            context = [{"role": "system", "content": SYSTEM_PROMPT}]
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
