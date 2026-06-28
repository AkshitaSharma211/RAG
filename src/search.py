import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "data/vector_store", collection_name: str = "pdf_documents", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.3-70b-versatile"):
        self.model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] RAGSearch initialized. Documents in store: {self.collection.count()}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        if not results['documents'] or not results['documents'][0]:
            return "No relevant documents found."
        context = "\n\n".join(results['documents'][0])
        prompt = f"""Use the following context to answer the question.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
        response = self.llm.invoke([prompt])
        return response.content

if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is video compression?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)