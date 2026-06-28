from src.data_loader import load_all_documents
from src.vectorstore import VectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    VECTOR_STORE_PATH = "data/vector_store"
    COLLECTION_NAME = "pdf_documents"
    
    docs = load_all_documents("data")
    store = VectorStore(COLLECTION_NAME, VECTOR_STORE_PATH)
    
    if store.collection.count() == 0:
        store.build_from_documents(docs)
    
    rag_search = RAGSearch(persist_dir=VECTOR_STORE_PATH, collection_name=COLLECTION_NAME)
    query = "What is video compression?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)