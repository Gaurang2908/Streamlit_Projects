# Pseudo GPT - Healthcare Analytics Assistant (Patched)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import ast
import io
import re

# OpenAI settings
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = "gpt-3.5-turbo"

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

        query_instruction = f"""
        You are a data assistant. Convert the following natural language question into a Python-readable dictionary with keys:
        - 'action': one of ['filter', 'plot']
        - 'filters': a list of pandas filter conditions as strings (if any)
        - 'plot_type': if action is 'plot', give one of ['bar', 'line', 'pie']
        - 'target_column': column to plot or count on

        Example output:
        {{'action': 'filter', 'filters': ["Age > 40", "Smoker == 'Yes'"], 'target_column': 'Smoker'}}

        Now parse: {prompt}
        """

        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You're a Python dictionary formatter."},
                    {"role": "user", "content": query_instruction}
                ]
            )
            parsed_text = completion.choices[0].message.content.strip()
            try:
                query = ast.literal_eval(parsed_text)
            except Exception:
                st.error("LLM returned invalid dictionary. Please try rephrasing.")
                st.stop()

            # Sanitize column names in filters
            def quote_column_names(filter_str, df_columns):
                for col in df_columns:
                    pattern = r'\\b{}\\b'.format(re.escape(col))
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
