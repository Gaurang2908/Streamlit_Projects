import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import ast
import re

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI Header
st.title("Pseudo GPT - Healthcare Analytics Assistant")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your healthcare dataset (CSV)", type="csv")
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

# OpenAI prompt helper to parse query into structured logic
def parse_query_with_llm(prompt, columns):
    system_prompt = f"""
    You are a data query assistant.
    Given a user prompt and list of CSV columns, return a Python dictionary with:
    - action: one of [filter, groupby, plot]
    - filters: list of conditions (e.g., Age > 40, Smoker == 'Yes')
    - plot_type: (optional) one of [bar, line, pie]
    - group_by: (optional) column name(s) to group by
    - target_column: (optional) column to aggregate or plot

    Use ONLY the columns from this list: {columns}
    Respond ONLY with the Python dictionary.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content.strip()

    try:
        parsed = ast.literal_eval(content)
        return parsed
    except Exception as e:
        return {"error": f"Could not parse response: {e}"}

# Main chatbot interface
prompt = st.chat_input("Ask your query (e.g. patients over 40 who smoke and were readmitted)")
if prompt and df is not None:
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append(("user", prompt))

    with st.chat_message("assistant"):
        columns = list(df.columns)
        parsed_query = parse_query_with_llm(prompt, columns)

        if "error" in parsed_query:
            st.error(parsed_query["error"])
        else:
            try:
                filters = parsed_query.get("filters", [])
                action = parsed_query.get("action")
                plot_type = parsed_query.get("plot_type")
                group_by = parsed_query.get("group_by")
                target_column = parsed_query.get("target_column")

                # Apply filters
                query_df = df.copy()
                for cond in filters:
                    query_df = query_df.query(cond)

                # Action: Filtered table
                if action == "filter":
                    st.success(f"Showing {len(query_df)} filtered records")
                    st.dataframe(query_df)

                # Action: Plot
                elif action == "plot" and plot_type:
                    fig, ax = plt.subplots()
                    if plot_type == "line":
                        if group_by and target_column:
                            query_df.groupby(group_by)[target_column].mean().plot(ax=ax)
                    elif plot_type == "bar":
                        if group_by and target_column:
                            query_df.groupby(group_by)[target_column].mean().plot(kind="bar", ax=ax)
                    elif plot_type == "pie":
                        if group_by:
                            query_df[group_by].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                    else:
                        st.warning("Unsupported plot type or missing fields.")
                        st.stop()
                    st.pyplot(fig)

                else:
                    st.warning("Unrecognized or missing action type.")

            except Exception as e:
                st.error(f"Failed to process your query: {e}")

    st.session_state.chat_history.append(("assistant", str(parsed_query)))

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
