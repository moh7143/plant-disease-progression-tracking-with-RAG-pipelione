import os
import json
import uuid
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from collections import defaultdict
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_COLLECTION = "rag_docs"
INDEX_FILE = "indexed_files.json"

def load_indexed_files(index_file=INDEX_FILE):
    if not os.path.exists(index_file):
        return set()
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data)
    except Exception:
        return set()

def save_indexed_files(files, index_file=INDEX_FILE):
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(sorted(list(files)), f, indent=4)

def get_new_files(folder_path):
    doc_dir = Path(folder_path)
    if not doc_dir.exists():
        raise FileNotFoundError(folder_path)
    indexed = load_indexed_files()
    all_files = list(doc_dir.glob("*.pdf")) + list(doc_dir.glob("*.txt"))
    new_files = []
    for f in all_files:
        if f.name not in indexed:
            new_files.append(f)
    return new_files

from langchain_community.document_loaders import TextLoader

def load_all_pdfs(folder_path):
    new_files = get_new_files(folder_path)
    if len(new_files) == 0:
        print("\nNo new PDFs or TXTs found.")
        print("Using existing ChromaDB embeddings.\n")
        return []
    all_docs = []
    print(f"\nFound {len(new_files)} NEW file(s).\n")
    for f in new_files:
        print(f"Loading {f.name}")
        try:
            if f.suffix.lower() == '.pdf':
                loader = PyMuPDFLoader(str(f))
            elif f.suffix.lower() == '.txt':
                loader = TextLoader(str(f), encoding="utf-8")
            else:
                continue
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = f.name
                doc.metadata["file_type"] = f.suffix.lower().replace('.', '')
                all_docs.append(doc)
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
    print(f"\nLoaded {len(all_docs)} pages/documents.\n")
    return all_docs

class RAGRetrieval:
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50,
        persist_dir="./chroma_db"
    ):
        print("\nInitializing ChromaDB & Loading AI Models...")
        print("-> Loading Embedding Model...")
        self.embed_model = SentenceTransformer(model_name)
        print("Embedding Model Loaded")

        print("-> Loading Reranker Model...")
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        print("Reranker Loaded")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print("-> Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=persist_dir)
        print("PersistentClient Created")

        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION
        )
        print("Collection Opened")

        try:
            total = self.collection.count()
            print(f"Collection contains {total} chunks")
        except Exception as e:
            print("Warning:", e)
    def rerank(self, query, chunks):
        if len(chunks) == 0:
            return []
        pairs = []
        for chunk in chunks:
            pairs.append((query, chunk["content"]))
        scores = self.reranker.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        chunks.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        return chunks

    def _get_embeddings(self, texts):
        return self.embed_model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False
        )

    def add_docs(self, documents):
        if len(documents) == 0:
            print("Nothing new to index.")
            return
        print(f"Documents received: {len(documents)}")
        empty_pages = 0
        for i, doc in enumerate(documents):
            if not doc.page_content.strip():
                empty_pages += 1
        print(f"Empty pages: {empty_pages}")
        print(f"Non-empty pages: {len(documents) - empty_pages}")
        if empty_pages != len(documents):
            for i, doc in enumerate(documents):
                if doc.page_content.strip():
                    print(f"\nFirst non-empty page: {i+1}")
                    print(doc.page_content[:500])
                    break
        if len(documents) > 0:
            print("First document preview:")
            chunks = self.splitter.split_documents(documents)
            print("Chunks:", len(chunks))
            print(documents[0].page_content[:500])
        print("\nSplitting documents into chunks...")
        chunks = self.splitter.split_documents(documents)
        if len(chunks) == 0:
            print("No chunks created.")
            return
        print(f"Created {len(chunks)} chunks.")
        texts = []
        ids = []
        metadatas = []
        indexed = load_indexed_files()
        for chunk in chunks:
            texts.append(chunk.page_content)
            ids.append(str(uuid.uuid4()))
            meta = dict(chunk.metadata)
            meta["content_length"] = len(chunk.page_content)
            metadatas.append(meta)
            indexed.add(meta["source_file"])
        print("Generating embeddings...")
        embeddings = self._get_embeddings(texts)
        print("Saving into ChromaDB...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        save_indexed_files(indexed)
        print("\nFinished indexing.")
        print("Collection size :", self.collection.count())

    def retrieve(
        self,
        query,
        search_k=10,
        chunks_per_document=2,
        final_k=3
    ):
        embedding = self._get_embeddings([query])
        results = self.collection.query(
            query_embeddings=embedding.tolist(),
            n_results=search_k
        )
        print("\n========== CHROMADB RESULTS ==========")
        print("Returned IDs:", len(results["ids"][0]))

        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            print(f"\nResult {i+1}")
            print("Distance:", dist)
            print(doc[:300])
        if len(results["ids"][0]) == 0:
            return []
        grouped = defaultdict(list)
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            grouped[meta["source_file"]].append({
                "content": doc,
                "metadata": meta,
                "distance": dist
            })
        merged = []
        for source in grouped:
            grouped[source].sort(
                key=lambda x: x["distance"]
            )
            merged.extend(
                grouped[source][:chunks_per_document]
            )
        reranked = self.rerank(query, merged)
        print("\n========== AFTER RERANK ==========")

        for i, chunk in enumerate(reranked[:10]):
            print(f"{i+1}. Score: {chunk['rerank_score']:.3f}")
            print(chunk["content"][:200])
        return reranked[:final_k]

    def total_documents(self):
        return self.collection.count()

    def reset(self):
        print("Deleting Chroma Collection...")
        self.client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION
        )
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        print("Collection reset completed.")