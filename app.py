
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
# from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
# from pandasai_app.components.faq import faq


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


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
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

uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are Support",
    on_change=clear_submit,
)

if uploaded_file:
    df = load_data(uploaded_file)

openai_api_key = st.sidebar.text_input("OpenAI API Key",
                                        type="password",
                                        placeholder="Paste your OpenAI API key here (sk-...)")

chat_data = st.sidebar.selectbox("Choose a Backend", ['langchain', 'pandasai'])

with st.sidebar:
        st.markdown("---")
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
            "2. Choose Backend PandasAI or LangChain\n"
            "3. Upload the file with data📄\n"
            "4. Ask a question about to make dataframe conversational💬\n"
        )

        st.markdown("---")

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if chat_data == "pandasai":
        #PandasAI OpenAI Model
        llm = OpenAI(api_token=openai_api_key)
        # llm = OpenAI(api_token=openai_api_key)

        sdf = SmartDataframe(df, config = {"llm": llm,
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
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
