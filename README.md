# RAG Pipeline — Chat with Your Notes

A Retrieval-Augmented Generation (RAG) pipeline that answers questions based on your PDF notes using ChromaDB and Groq LLM.

## What it does
Upload your PDF notes, ask any question, and get accurate answers with citations showing exactly which page the answer came from.

## Tech Stack
- **Document Loading** — PyMuPDF
- **Text Splitting** — LangChain RecursiveCharacterTextSplitter
- **Embeddings** — SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Store** — ChromaDB
- **LLM** — Groq (LLaMA 3.3 70b)

## Setup

1. Clone the repo
```bash
git clone https://github.com/AkshitaSharma211/RAG.git
cd RAG
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your Groq API key
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

4. Run the notebook
Open `pdf_loader.ipynb` and run all cells top to bottom.

## How it works
1. PDFs are loaded and split into chunks
2. Each chunk is converted to embeddings (384-dim vectors)
3. Embeddings stored in ChromaDB
4. User question is embedded and matched against stored chunks
5. Top matching chunks sent to Groq LLM for answer generation

## Example
```python
result = adv_rag.query("What is video compression?", top_k=3, min_score=0.1)
print(result['answer'])
```