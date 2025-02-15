# RAG AI Tutorial

This repository demonstrates how to use **Retrieval-Augmented Generation (RAG)** to enhance LLM responses by retrieving **external knowledge** using **Ollama, ChromaDB, and LangChain**.

## Installation

### Install Ollama

Ollama allows you to run LLMs locally. Download and install it from:

**➡ [Ollama Website](https://ollama.com/)**

After installation, verify that Ollama is installed by running:

    ollama --version

### Download Llama 3.2:1b and nomic-embed-text Models

Run the following command to download the models:

    ollama pull llama3.2:1b
    ollama pull nomic-embed-text

### Set Up a Virtual Environment

It is recommended to use a virtual environment for Python dependencies.

#### macOS/Linux:

    python3 -m venv venv
    source venv/bin/activate

#### Windows (Powershell):

    python -m venv venv
    venv\Scripts\activate

### Install Required Packages

Once inside the virtual environment, install the required dependencies:

    pip install ollama langchain langchain_community langchain-ollama chromadb

---

## Usage

Run the script

    python app.py

## How It Works

1. The script first **queries the LLM without RAG**.
2. Then, it **stores an external document** (`openai_deep_research.txt`) in **ChromaDB**.
3. When asked again, the model **retrieves relevant context** before generating a response.
4. The final output **corrects the original mistakes**, showing the power of RAG.
