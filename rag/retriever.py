"""
voice_agent/rag/retriever.py

ChromaDB-based semantic retrieval system for the voice agent.
Uses sentence-transformers to create embeddings and ChromaDB to store/search them.
Index persists on disk — documents are only re-embedded when new files are added.
Retrieval uses cosine similarity, so it handles typos, synonyms, and paraphrasing.
"""

import os
import time
import hashlib
import json
import chromadb
from rag.document_loader import Chunk, load_directory


# Persistent DB and metadata paths
_CHROMA_DIR = "./chroma_db"
_META_FILE  = os.path.join(_CHROMA_DIR, "indexed_files.json")


def _compute_dir_hash(directory: str) -> str:
    """Hash filenames + sizes to detect when documents change."""
    entries = []
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            entries.append(f"{fname}:{os.path.getsize(fpath)}")
    return hashlib.md5("|".join(entries).encode()).hexdigest()


def _needs_reindex(directory: str) -> bool:
    """Check if the documents folder has changed since last index."""
    current_hash = _compute_dir_hash(directory)
    if os.path.exists(_META_FILE):
        with open(_META_FILE, "r") as f:
            saved = json.load(f)
        if saved.get("hash") == current_hash:
            return False
    return True


def _save_index_meta(directory: str):
    """Save the current state of the documents folder."""
    os.makedirs(_CHROMA_DIR, exist_ok=True)
    with open(_META_FILE, "w") as f:
        json.dump({"hash": _compute_dir_hash(directory)}, f)


class RAGRetriever:
    def __init__(self, documents_dir: str):
        t0 = time.time()
        self.documents_dir = documents_dir

        # Connect to persistent ChromaDB
        self.client = chromadb.PersistentClient(path=_CHROMA_DIR)

        if _needs_reindex(documents_dir):
            print("[RAG] Indexing documents (first time or files changed)...")
            self._build_index()
        else:
            self.collection = self.client.get_collection("voice_agent_docs")
            count = self.collection.count()
            print(f"[RAG] Loaded existing index: {count} chunks (no re-indexing needed)")

        # Also keep chunks in memory for source attribution
        self.chunks = load_directory(documents_dir)
        print(f"[RAG] Ready in {time.time()-t0:.2f}s")

    def _build_index(self):
        """Parse documents, embed them, and store in ChromaDB."""
        # Delete old collection if it exists
        try:
            self.client.delete_collection("voice_agent_docs")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name="voice_agent_docs",
            metadata={"hnsw:space": "cosine"},
        )

        chunks = load_directory(self.documents_dir)
        if not chunks:
            print("[RAG] No documents found. RAG disabled.")
            return

        # Add chunks in batches (ChromaDB limit is ~5000 per batch)
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                documents=[c.text for c in batch],
                metadatas=[{"source_file": c.source_file, "location": c.location} for c in batch],
                ids=[f"chunk_{i+j}" for j in range(len(batch))],
            )

        _save_index_meta(self.documents_dir)
        print(f"[RAG] Indexed {len(chunks)} chunks into ChromaDB")

    def retrieve(self, query: str, top_k: int = 3) -> tuple[str, list[dict]]:
        """
        Retrieve the top-k most relevant chunks for a query using semantic search.

        Returns:
            context_str: formatted string to inject into the LLM prompt
            sources: list of {file, location, snippet} dicts for logging
        """
        if not self.collection or self.collection.count() == 0:
            return "", []

        t0 = time.time()

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        if not results["documents"] or not results["documents"][0]:
            return "", []

        sources = []
        context_parts = []

        for rank, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0]), 1
        ):
            context_parts.append(
                f"[Source {rank}: {meta['source_file']}, {meta['location']}]\n{doc}"
            )
            sources.append({
                "file": meta["source_file"],
                "location": meta["location"],
                "snippet": doc[:80] + "..." if len(doc) > 80 else doc,
            })

        context_str = "\n\n".join(context_parts)
        elapsed_ms = (time.time() - t0) * 1000
        print(f"[RAG] Retrieved {len(sources)} chunks in {elapsed_ms:.1f}ms")

        return context_str, sources
