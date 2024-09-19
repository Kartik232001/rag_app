from io import BytesIO
import os
import time 
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
import streamlit as st

from utils.models import fetch_models
from utils.models import embed_model_install

models = fetch_models()

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

# Function to handle the PDF upload
def handle_pdf_upload(model):
    pdf_file = st.file_uploader("Upload your PDF file", type="pdf")
    if pdf_file is not None:
        # Save the uploaded PDF to the session state
        st.session_state['pdf_file'] = pdf_file
        st.session_state['uploaded'] = True
        # st.success("PDF uploaded successfully!")
        with st.spinner(text="Processing the docs..."):
            rag_chain = rag_init(model)
        st.write("Click on the 'Chat' page to continue.")
    else:
        if 'uploaded' in st.session_state and st.session_state['uploaded']:
            st.success("PDF already uploaded. You can proceed to the chat.")

def stream_response(question: str):
    response = st.session_state['rag_chain'].invoke({"input": question})
    st.session_state.messages.append({"role": "system", "content": response["answer"]})
    for word in response["answer"].split():
        yield word + " "
        time.sleep(0.05)

# Function to handle the chat interface
def handle_chat_interface():
    if 'pdf_file' in st.session_state and st.session_state['uploaded']:

        st.title(f"Chat with {st.session_state['pdf_file'].name.split('.')[0]}")

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if question := st.chat_input("enter your query"):
            st.session_state.messages.append({"role": "human", "content": question})

            with st.chat_message("human"):
                st.write(question)

            # with st.spinner(text="Generating response..."):
            #     response = st.session_state['rag_chain'].invoke({"input": question})

            with st.chat_message("system"):
                st.write(stream_response(question))
            
            # st.session_state.messages.append({"role": "system", "content": response["answer"]})
    else:
        st.warning("Please upload a PDF before accessing the chat interface.")


# Main function to switch between pages
def main():
    """Main function to run the app."""
    st.sidebar.title("Navigation")
    model = st.sidebar.selectbox("Select model", models)
    page = st.sidebar.radio("Go to", ["PDF Upload", "Chat"])

    st.sidebar.text(" ")
    st.sidebar.button("Reset chat", on_click=lambda: st.session_state['messages'].clear())
    st.sidebar.text(" ")
    st.sidebar.button("Reset PDF", on_click=lambda: st.session_state.clear())
    
    if page == "PDF Upload":
        handle_pdf_upload(model)
    elif page == "Chat":
        handle_chat_interface()

# Initialize session state
if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = False

# Run the app
if __name__ == "__main__":
    # if 'uploaded' not in st.session_state:
    #     st.session_state['uploaded'] = False
    # if 'messages' not in st.session_state:
    #     st.session_state['messages'] = []
    main()