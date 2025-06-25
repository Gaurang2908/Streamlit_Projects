import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import ast
import io
import re
import difflib

# OpenAI API Client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = "gpt-3.5-turbo"

# Scope-defined Column Definitions (from analytics assistant)
COLUMN_DEFINITIONS = {
    "Patient ID": "Unique ID associated with each patient",
    "Age": "Age of the individual, ranging from 20 to 85",
    "Gender": "(Here)M/F",
    "BMI": "BMI is a tool that estimates the amount of body fat by using height and weight measurements. It can help assess health risks, but it has some limitations and does not diagnose conditions. Calculated using BMI = (Weight)/(Height in meters)^2",
    "Smoker": "Indicates if the patient is a smoker (Yes/No)",
    "Diabetes": "Indicates if the patient has diabetes (Yes/No)",
    "Exercise Frequency (Days/week)": "Number of days a patient exercises in a week (out of 7) - ranges from 0 to 7",
    "Cholesterol level": "Cholesterol levels show how much cholesterol is circulating in your blood.(LDL + HDL) 100 to 200 (usual range for metric)",
    "Hospital visit data (for last year)": "Number of hospital visits by the patient in the last year",
    "Readmissions within 30 days": "If the patient was readmitted to the hospital within 30 days of being discharged"
}

# Helper: Match fuzzy columns
def get_closest_column(colname, columns):
    matches = difflib.get_close_matches(colname, columns, n=1, cutoff=0.6)
    return matches[0] if matches else None

# App UI
st.set_page_config(page_title="Pseudo GPT v2", layout="wide")
st.title("Pseudo GPT v2 - Healthcare Data Assistant")

uploaded_file = st.file_uploader("Upload your healthcare dataset (CSV)", type=["csv"])
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    st.subheader("Data Preview")
    st.dataframe(df.head())

# Chat Interface
st.subheader("Conversation")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a healthcare-related data question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if df is None:
        response = "Please upload a dataset first."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        column_docs = "\n".join([f"- {col}: {desc}" for col, desc in COLUMN_DEFINITIONS.items() if col in df.columns])
        columns_str = ", ".join([f"'{col}'" for col in df.columns])

        query_instruction = f"""
        You are a healthcare analytics assistant. You help users interpret and analyze healthcare-related data from CSV files.

        Here are the column definitions for reference:
        {column_docs}

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

        For queries like:
        - "patients over 40 who smoke" → filters: ["Age > 40", "Smoker == 'Yes'"]
        - "bar chart for diabetes vs smoker count" → action: plot, plot_type: bar, target_column: 'Diabetes'

        Output format:
        {{
            "action": "filter" or "plot",
            "filters": [...],
            "plot_type": "bar" | "pie" | "line" | "scatter" | "hist" | "area",
            "target_column": "ColumnName"
        }}

        Your response must be a valid Python dictionary. Do not include explanations.
        Instruction: {prompt}
        """

        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You're a Python dictionary generator for data queries."},
                    {"role": "user", "content": query_instruction}
                ]
            )
            response_text = completion.choices[0].message.content.strip()
            try:
                query = ast.literal_eval(response_text)
            except:
                st.error("LLM response was not a valid dictionary.")
                st.stop()

            def quote_columns(expr, cols):
                for col in cols:
                    expr = re.sub(rf'\b{re.escape(col)}\b', f'`{col}`', expr)
                return expr

            result = df.copy()
            for f in query.get("filters", []):
                safe_filter = quote_columns(f, df.columns)
                try:
                    result = result.query(safe_filter)
                except Exception as e:
                    st.warning(f"Filter failed: `{safe_filter}` — {e}")

            output_buffer = io.StringIO()
            if query["action"] == "filter":
                response = f"Filtered {len(result)} rows based on your query."
                st.dataframe(result)

            elif query["action"] == "plot":
                col = query.get("target_column")
                plot_type = query.get("plot_type", "bar")
                col = get_closest_column(col, df.columns) or col

                if col not in result.columns:
                    response = f"'{col}' not found in dataset."
                elif result[col].dropna().empty:
                    response = f"No data to plot in '{col}'."
                else:
                    fig, ax = plt.subplots()
                    counts = result[col].value_counts().sort_index()

                    if plot_type == "bar":
                        counts.plot(kind="bar", ax=ax)
                    elif plot_type == "pie":
                        counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
                        ax.axis("equal")
                    elif plot_type == "line":
                        counts.plot(kind="line", marker="o", ax=ax)
                    elif plot_type == "hist":
                        result[col].dropna().plot(kind="hist", bins=10, ax=ax)
                    elif plot_type == "scatter":
                        if len(result.columns) >= 2:
                            x_col = result.columns[0]
                            result.plot(kind="scatter", x=x_col, y=col, ax=ax)
                        else:
                            st.warning("Need at least 2 numeric columns for scatter plot.")
                    elif plot_type == "area":
                        counts.plot(kind="area", stacked=False, ax=ax)

                    ax.set_title(f"{plot_type.title()} chart for {col}")
                    st.pyplot(fig)
                    response = f"Plotted a {plot_type} chart for {col}."

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            response = f"Error: {e}"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
