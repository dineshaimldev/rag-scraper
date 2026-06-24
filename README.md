# RAG Chat

A simple chat application that lets you paste any article URL and ask questions about it. It uses local AI models via Ollama so everything runs on your machine.

---

## Requirements

- Python 3.9 or higher
- Ollama installed and running

---

## Installation

Install the required Python packages:

```bash
pip install streamlit langchain langchain-community langchain-text-splitters langchain-chroma langchain-ollama chromadb beautifulsoup4
```

Install Ollama from https://ollama.com and pull the required models:

```bash
ollama pull all-minilm
ollama pull llama3.2:1b
```

---

## Running the App

```bash
streamlit run Main.py
```

The app will open in your browser at http://localhost:8501

---

## How to Use

1. Open the app in your browser
2. Paste any article URL in the chat input and press Enter
3. Wait for the article to load and the vector store to build
4. Ask any question about the article
5. To switch to a different article, paste a new URL at any time

---

## Project Structure

```
rag-scraper/
    Main.py         application code
    chroma_db/      vector store saved here after first run
    README.md
```

---

## Notes

- The embeddings are saved to the chroma_db folder so they persist between sessions
- Ollama must be running in the background before starting the app
- The app only answers questions based on the content of the loaded article
