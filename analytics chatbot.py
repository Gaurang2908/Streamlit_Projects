import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import ast
import io
import re
import difflib

# OpenAI settings
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = "gpt-3.5-turbo"

# Column Definitions
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

# App UI
st.title("Analytics Healthcare Assistant")

uploaded_file = st.file_uploader("Upload your healthcare dataset (CSV)", type=["csv"])
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # LLM interaction
    prompt = st.chat_input("Write something...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # Compose data and definitions for LLM
        columns_str = ", ".join([f"'{col}'" for col in df.columns])
        column_docs = "\n".join([f"- {col}: {desc}" for col, desc in COLUMN_DEFINITIONS.items() if col in df.columns])

        query_instruction = f"""
        You are a healthcare analytics assistant. You work with tabular CSV data uploaded by the user.
        You must always refer only to these columns:
        {columns_str}

        Descriptions:
        {column_docs}

        Examples:
        - "patients over 40 who smoke" → filters: ["Age > 40", "Smoker == 'Yes'"]
        - "pie chart of diabetes" → action: plot, plot_type: pie, target_column: 'Diabetes'

        Required Output:
        {{
            "action": "filter" or "plot",
            "filters": [...],
            "plot_type": "bar"|"pie"|"line",
            "target_column": "ColumnName"
        }}

        Only return a valid Python dictionary and nothing else.
        Instruction: {prompt}
        """

        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You're a precise Python dictionary generator."},
                    {"role": "user", "content": query_instruction}
                ]
            )
            parsed_text = completion.choices[0].message.content.strip()
            try:
                query = ast.literal_eval(parsed_text)
            except Exception:
                st.error("LLM returned invalid dictionary. Please try rephrasing.")
                st.stop()

            def quote_column_names(filter_str, df_columns):
                for col in df_columns:
                    pattern = r'\b{}\b'.format(re.escape(col))
                    filter_str = re.sub(pattern, f"`{col}`", filter_str)
                return filter_str

            # Filter logic
            if query['action'] == 'filter':
                result = df.copy()
                for f in query.get("filters", []):
                    safe_filter = quote_column_names(f, df.columns)
                    try:
                        result = result.query(safe_filter)
                    except Exception as e:
                        st.warning(f"Filter failed: `{safe_filter}` — {e}")
                st.markdown(f"### Filtered Result ({len(result)} rows)")
                st.dataframe(result)

            # Plot logic
            elif query['action'] == 'plot':
                result = df.copy()
                for f in query.get("filters", []):
                    safe_filter = quote_column_names(f, df.columns)
                    try:
                        result = result.query(safe_filter)
                    except Exception as e:
                        st.warning(f"Filter failed: `{safe_filter}` — {e}")

                col = query.get("target_column")
                plot_type = query.get("plot_type", "bar")

                closest = get_closest_column(col, df.columns)
                if closest and closest != col:
                    col = closest

                if col not in result.columns:
                    st.warning("Column not found in dataset.")
                elif result[col].dropna().empty:
                    st.warning("No valid data to plot.")
                else:
                    fig, ax = plt.subplots()
                    counts = result[col].dropna().value_counts()

                    if plot_type == 'bar':
                        counts.plot(kind='bar', ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_title(f"Bar chart for {col}")
                    elif plot_type == 'pie':
                        counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                        ax.set_ylabel("")
                        ax.set_title(f"Pie chart for {col}")
                        ax.axis('equal')
                    elif plot_type == 'line':
                        counts.sort_index().plot(kind='line', ax=ax)
                        ax.set_ylabel("Count")
                        ax.set_title(f"Line chart for {col}")

                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Could not process the request: {e}")
