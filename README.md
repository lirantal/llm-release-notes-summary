# Node.js release notes summaries with LLMs

This repository contains the code for an example project that uses the following tools and techniques to summarize Node.js version release notes with Large Language Models (LLM):

- Python
- LangChain
- LangChain Prompt Template usage
- LangChain DocumentsLoader with WebBaseLoader
- LangChain RAG Pipelines
- Hugging Face Transformers for local embeddings
- Chroma for in-memory vector database
- OpenAI integration for LLM inference

## How to run

### 1. Clone the project

```bash
git clone https://github.com/lirantal/llm-release-notes-summary
```

### Create a Python virtual environment

Change into the project's cloned directory and create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Run the project

Run the project:

```bash
python main.py
```
