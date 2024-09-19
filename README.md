# 🔍 RAG (Retrieval-Augmented Generation) Application with Streamlit & Ollama

Welcome to the **RAG Application**! This project is designed to demonstrate the power of **local LLMs** using Ollama, a **Streamlit** interface, and the retrieval and generation capabilities of **Langchain**.

Here, you can upload a pdf document on the application and then proceed to inquire about anything related to the uploaded document. 

## 🚀 Features

- **RAG Workflow**: Combines local knowledge retrieval with advanced generation models.
- **Local LLM with Ollama**: Run large language models locally, avoiding the need for cloud-based APIs.
- **Streamlit UI**: Interactive and user-friendly web interface to explore RAG workflows.
- **Langchain Integration**: Seamless orchestration of retrieval and generation pipelines.
- **Customizable**: Adapt the system to your own datasets, models, and needs.

## 🛠️ Requirements

Before running the application, make sure you have the following dependencies installed:

1. **Python 3.8+**
2. **Streamlit**: Install via `pip install streamlit`
3. **Ollama**: Installed and running on your machine for local LLM inference.
4. **Langchain**: Install via `pip install langchain`

### Ollama Installation

- Follow the instructions in the [Ollama documentation](https://ollama.com/) to install and set up.
- Make sure ollama is running locally.

## 📦 Installation

1. Clone this repository:

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   ```
   For Mac
   ```bash
   source venv/bin/activate
   ```
   For Windows
   ```bash
   venv\scripts\activate.bat
   ```
3. Install the packages

   ```bash
   pip3 install -r requirements.txt
   ```

## 🚀 Running the Application

1. Make sure **Ollama** is running locally.
2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```
   

