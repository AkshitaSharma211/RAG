# RAG 🔍

A Retrieval-Augmented Generation (RAG) system that combines document retrieval with AI-powered response generation.

---

## 📌 Overview
This project implements a RAG pipeline that:
- Retrieves relevant documents from a knowledge base
- Generates accurate, context-aware responses using an LLM

---

## 🛠️ Tech Stack
- Python
- LangChain / LlamaIndex
- OpenAI / Gemini / Ollama
- FAISS / ChromaDB (Vector Store)

---

## ⚙️ Installation

```bash
git clone https://github.com/AkhitaSharma211/RAG.git
cd RAG
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python main.py
```

---

## 📁 Project Structure

RAG/
├── main.py
├── requirements.txt
├── data/
├── embeddings/
└── README.md
## Progress
- Document Loading (TextLoader, DirectoryLoader, PyPDFLoader)
- Text Splitting (RecursiveCharacterTextSplitter)
- Embeddings (all-MiniLM-L6-v2, 384 dimensions)
