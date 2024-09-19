import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.app_logo import add_logo

import time
import os 
from io import BytesIO

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama

from utils.models import fetch_models


models = fetch_models()

# Set page config
st.set_page_config(page_title="PDF Chat Assistant", layout="wide")

st.markdown("""
<style>
    # .stApp {
    #     background-color: #f0f2f6;
    # }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox {
        border-radius: 5px;
    }
    /* Sidebar styling */
    .css-1n76uvr {
        background-color: #2C3E50;
        color: #ECF0F1;
    }
    .css-1n76uvr .stSelectbox label,
    .css-1n76uvr .stRadio label {
        color: #ECF0F1;
    }
    .css-1n76uvr .stSelectbox > div > div,
    .css-1n76uvr .stRadio > div {
        background-color: #34495E;
        color: #ECF0F1;
        border: none;
    }
    .css-1n76uvr .stButton>button {
        background-color: #E74C3C;
    }
    .css-1n76uvr .stButton>button:hover {
        background-color: #C0392B;
    }
    /* Custom divider */
    .sidebar-divider {
        margin-top: 20px;
        margin-bottom: 20px;
        border-top: 1px solid #7F8C8D;
    }
</style>
""", unsafe_allow_html=True)


def rag_init(model):

    print("**********************************")
    print("inside rag init")
    print(f"model: {model}")
    print("**********************************")

    llm = ChatOllama(
        model=model
    )

    embedder = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    docs = []

    directory = "directory"
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open("directory/temp.pdf", "wb") as f:
        f.write(st.session_state['pdf_file'].getvalue())
    st.toast(body="upload successful!")

    loader = PyMuPDFLoader(f"directory/temp.pdf")
    docs += loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embedder)

    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. Always respond politely to any greetings. "
        "If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer under 30 words."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    st.session_state['rag_chain'] = rag_chain

    os.remove("directory/temp.pdf")

    return rag_chain


def handle_pdf_upload(model):
    st.markdown("## 📄 Upload Your PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    if pdf_file is not None:
        st.session_state['pdf_file'] = pdf_file
        st.session_state['uploaded'] = True
        with st.spinner("Processing the document..."):
            rag_chain = rag_init(model)
        st.success("✅ PDF uploaded and processed successfully!")
        st.info("👉 Navigate to the 'Chat' page to start your conversation.")
    else:
        if 'uploaded' in st.session_state and st.session_state['uploaded']:
            st.success("✅ PDF already uploaded. You can proceed to the chat.")

def stream_response(question: str):
    response = st.session_state['rag_chain'].invoke({"input": question})
    st.session_state.messages.append({"role": "system", "content": response["answer"]})
    for word in response["answer"].split():
        yield word + " "
        time.sleep(0.05)

def handle_chat_interface():
    if 'pdf_file' in st.session_state and st.session_state['uploaded']:
        colored_header(label=f"💬 Chat with {st.session_state['pdf_file'].name.split('.')[0]}", 
                       description="Ask questions about your PDF",
                       color_name="green-70")

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if question := st.chat_input("Ask a question about your PDF"):
            st.session_state.messages.append({"role": "human", "content": question})
            with st.chat_message("human"):
                st.write(question)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in stream_response(question):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
    else:
        st.warning("⚠️ Please upload a PDF before accessing the chat interface.")
        st.info("👈 Go to the 'PDF Upload' page to upload your document.")


def main():

    st.sidebar.markdown("<h1 style='text-align: center; color: #ECF0F1;'>📚 PDF Chat Assistant</h1>", unsafe_allow_html=True)
    add_vertical_space(2)
    
    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🤖 Choose Model")
    model = st.sidebar.selectbox("Select your preferred model", models, key="model_select")
    add_vertical_space(1)
    
    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📌 Navigation")
    page = st.sidebar.radio("Choose a page", ["PDF Upload", "Chat"], key="page_select")
    add_vertical_space(2)
    
    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Reset Chat", key="reset_chat"):
            st.session_state['messages'] = []
            st.success("Chat history cleared!")
    with col2:
        if st.button("🗑️ Reset PDF", key="reset_pdf"):
            st.session_state.clear()
            st.success("PDF and chat history cleared!")

    if page == "PDF Upload":
        handle_pdf_upload(model)
    elif page == "Chat":
        handle_chat_interface()


if __name__ == "__main__":
    main()