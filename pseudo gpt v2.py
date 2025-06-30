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

# Column Definitions (from Pseudo GPT)
COLUMN_DEFINITIONS = {
    "Patient ID": "Unique identifier for each patient",
    "Age": "Patient's age in years",
    "Gender": "Patient's gender (M/F)",
    "Smoker": "Yes/No field indicating if the patient is a smoker",
    "Diabetes": "Yes/No field indicating if the patient has diabetes",
    "BMI": "Body Mass Index calculated as weight(kg)/height(m)^2",
    "Exercise Frequency (Days/week)": "How many days per week the patient exercises (0-7)",
    "Cholesterol level": "Total cholesterol level (HDL + LDL)",
    "Hospital visit data (for last year)": "Number of hospital visits in the past year",
    "Readmissions within 30 days": "Yes/No for whether patient was readmitted within 30 days"
}

# Helper: Match fuzzy columns
def get_closest_column(colname, columns):
    matches = difflib.get_close_matches(colname, columns, n=1, cutoff=0.6)
    return matches[0] if matches else None

# Helper: Quote columns in filters
def quote_columns(expr, cols):
    for col in cols:
        expr = re.sub(rf'\b{re.escape(col)}\b', f'`{col}`', expr)
    return expr

# App UI
st.set_page_config(page_title="Pseudo GPT v2", layout="wide")
st.title("Pseudo GPT v2")

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
        You are a helpful and accurate healthcare analytics assistant.
        You analyze uploaded CSV data using only the following columns:
        {columns_str}

        Column meanings:
        {column_docs}

        For queries like:
        - "patients over 40 who smoke" → filters: ["Age > 40", "Smoker == 'Yes'"]
        - "bar chart for diabetes vs smoker count" → action: plot, plot_type: bar, target_column: 'Diabetes'

        If the query is general like "what is healthcare", answer directly in plain English.

        Output format:
        {{
            "action": "filter" or "plot" or "answer",
            "filters": [...],
            "plot_type": "bar" | "pie" | "line" | "scatter" | "hist" | "area",
            "target_column": "ColumnName",
            "answer": "Only if action is 'answer'"
        }}

        Your response must be a valid Python dictionary. Do not include explanations.
        Instruction: {prompt}
        """

        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You're a Python dictionary generator for data queries. Always respond with a Python dictionary."},
                    {"role": "user", "content": query_instruction}
                ]
            )

            response_text = completion.choices[0].message.content.strip()
            query = ast.literal_eval(response_text)

            result = df.copy()
            for f in query.get("filters", []):
                if any(op in f for op in ["==", "!=", ">", "<", ">=", "<="]):
                    safe_filter = quote_columns(f, df.columns)
                    try:
                        result = result.query(safe_filter)
                    except Exception as e:
                        st.warning(f"Filter failed: `{safe_filter}` — {e}")
                else:
                    st.warning(f"Ignored invalid filter (likely definition): `{f}`")

            if query["action"] == "plot":
                col = query.get("target_column")
                plot_type = query.get("plot_type", "bar")
                col = get_closest_column(col, df.columns) or col

                if col not in result.columns:
                    response = f"'{col}' not found in dataset."
                elif result[col].dropna().empty:
                    response = f"No data to plot in '{col}'."
                else:
                    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # smaller figure
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
                    fig.tight_layout()
                    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                    st.pyplot(fig, use_container_width=False)
                    st.markdown("</div>", unsafe_allow_html=True)
                    response = f"Plotted a {plot_type} chart for {col}."

            elif query["action"] == "filter":
                response = f"Filtered {len(result)} rows based on your query."
                st.dataframe(result)

            elif query["action"] == "answer":
                response = query.get("answer", "I don't know.")

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            response = f"Error: {e}"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
