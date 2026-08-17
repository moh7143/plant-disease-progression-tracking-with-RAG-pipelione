
from document_loader import load_all_pdfs, RAGRetrieval
from llm_service import OllamaLLM
retriever = None
llm = None
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
from document_loader import load_all_pdfs, RAGRetrieval
from llm_service import OllamaLLM
def initialize_rag():
    global retriever
    global llm
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(BASE_DIR, "textfiles")
    docs = load_all_pdfs(folder_path)
    retriever = RAGRetrieval(
        chunk_size=400,
        chunk_overlap=16
    )
    retriever.add_docs(docs)
    llm = OllamaLLM(
        model_name="qwen2.5:3b"
    )
    print("RAG Loaded Successfully")
def print_sources(contexts):
    if len(contexts) == 0:
        return
    print("\n===================================")
    print("Sources Used")
    print("================================")
    shown = set()
    for doc in contexts:
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown")
        if source not in shown:
            shown.add(source)
            print(f"• {source}")
    print("==============================\n")
def get_disease_details(query, search_query=None):
    if not search_query:
        search_query = query
    contexts = retriever.retrieve(
    query=search_query,
    search_k=5,
    chunks_per_document=2,
    final_k=3
)
    if len(contexts) == 0:
        return "No information found."
    answer = llm.generate_answer(
        query=query,
        contexts=contexts
    )
    return answer
def main():
    folder_path = r"C:\Users\Dell\Music\models\rag practices\textfiles"
    print("\n===================================")
    print("Plant Disease RAG System")
    print("=========================\n")
    
    print("Step 1 : Loading PDFs")
    docs = load_all_pdfs(folder_path)
    
    print("\nStep 2 : Loading ChromaDB & Retrieval Models")
    retriever = RAGRetrieval(
        chunk_size=500,
        chunk_overlap=50
    )
    
    if len(docs) > 0:
        print("\nStep 3 : Indexing New PDFs")
        retriever.add_docs(docs)
    else:
        print("\nStep 3 : No new PDFs detected.")
        
    print("\nStep 4 : Loading Ollama LLM")
    llm = OllamaLLM(
        model_name="qwen2.5:3b"
    )
    if not llm.test_connection():
        print("\nCould not connect to Ollama. Make sure Ollama app/service is running.")
        return
    
if __name__ == "__main__":
    main()