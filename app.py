# from langchain.llms import OpenAI
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.callbacks import StdoutCallback
from pandasai.llm import OpenAI
import streamlit as st
import pandas as pd
import os

from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def clear_submit():
    """
    Clear the Submit Button State
    Returns:
    """
    st.session_state["submit"] = False

# Function to process a single uploaded file
def process_uploaded_file(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

st.set_page_config(page_title="Chat-Data", page_icon="🦜|🐼")
st.title("Q&A with Data with AI 🐼")

# Prompt user for OpenAI API key
openai_api_key = st.sidebar.text_input("OpenAI API Key",
                                       type="password",
                                       placeholder="Paste your OpenAI API key here (sk-...)")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# Use file_uploader in a loop to handle multiple files
uploaded_files = st.file_uploader(
    label="Upload Data files",
    type=list(file_formats.keys()),
    accept_multiple_files=True,
    help="Various File formats are Support",
    on_change=clear_submit,
)

# Check if there are uploaded files before processing
if uploaded_files:
    # Combine uploaded files into one vector (concatenate along axis 0)
    combined_df = pd.concat([process_uploaded_file(file) for file in uploaded_files], axis=0, ignore_index=True)
else:
    combined_df = pd.DataFrame()  # Create an empty dataframe

chat_data = st.sidebar.selectbox("Choose a Backend", ['langchain', 'pandasai'])

# Process combined dataframe only if there are files uploaded
if not combined_df.empty:
    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What is this data about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if chat_data == "pandasai":
            # PandasAI OpenAI Model
            llm = OpenAI(api_token=openai_api_key)

            sdf = SmartDataframe(combined_df, config={"llm": llm,
                                                      "enable_cache": False,
                                                      "conversational": True,
                                                      "callback": StdoutCallback()})

            with st.chat_message("assistant"):
                response = sdf.chat(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)

        if chat_data == "langchain":
            llm = ChatOpenAI(
                temperature=0, model="gpt-3.5-turbo-1106", openai_api_key=openai_api_key, streaming=True
            )

            pandas_df_agent = create_pandas_dataframe_agent(
                llm,
                combined_df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
            )

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
else:
    st.warning("No files uploaded. Please upload data files to continue.")
