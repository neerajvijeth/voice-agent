"""
voice_agent/rag/retriever.py

PostgreSQL + pgvector semantic retrieval system for the voice agent.
Uses sentence-transformers to create embeddings and pgvector for similarity search.

Index persists in PostgreSQL — documents are only re-embedded when new files are added.
Retrieval uses cosine similarity with a configurable threshold to filter irrelevant chunks.
"""

import os
import time
import hashlib
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from rag.document_loader import Chunk, load_directory
from config import (
    PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD,
    RAG_EMBEDDING_MODEL, RAG_SIMILARITY_THRESHOLD,
)


class RAGRetriever:
    def __init__(self, documents_dir: str):
        t0 = time.time()
        self.documents_dir = documents_dir

        # Load embedding model (runs on GPU if available)
        print(f"[RAG] Loading embedding model '{RAG_EMBEDDING_MODEL}'...")
        self.embed_model = SentenceTransformer(RAG_EMBEDDING_MODEL)
        self.embed_dim = self.embed_model.get_embedding_dimension()

        # Connect to PostgreSQL
        self.conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DBNAME,
            user=PG_USER,
            password=PG_PASSWORD,
        )
        register_vector(self.conn)
        self._ensure_schema()

        # Check if re-indexing is needed
        if self._needs_reindex():
            print("[RAG] Indexing documents (first time or files changed)...")
            self._build_index()
        else:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            count = cur.fetchone()[0]
            cur.close()
            print(f"[RAG] Loaded existing index: {count} chunks (no re-indexing needed)")

        print(f"[RAG] Ready in {time.time()-t0:.2f}s")

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        cur = self.conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                source_file TEXT NOT NULL,
                location TEXT NOT NULL,
                embedding vector({self.embed_dim}) NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS index_metadata (
                id INTEGER PRIMARY KEY DEFAULT 1,
                dir_hash TEXT NOT NULL
            )
        """)
        self.conn.commit()
        cur.close()

    def _compute_dir_hash(self) -> str:
        """Hash filenames + sizes to detect when documents change."""
        entries = []
        for fname in sorted(os.listdir(self.documents_dir)):
            fpath = os.path.join(self.documents_dir, fname)
            if os.path.isfile(fpath):
                entries.append(f"{fname}:{os.path.getsize(fpath)}")
        return hashlib.md5("|".join(entries).encode()).hexdigest()

    def _needs_reindex(self) -> bool:
        """Check if the documents folder has changed since last index."""
        current_hash = self._compute_dir_hash()
        cur = self.conn.cursor()
        cur.execute("SELECT dir_hash FROM index_metadata WHERE id = 1")
        row = cur.fetchone()
        cur.close()
        if row and row[0] == current_hash:
            return False
        return True

    def _save_index_meta(self):
        """Save the current state of the documents folder."""
        cur = self.conn.cursor()
        current_hash = self._compute_dir_hash()
        cur.execute("""
            INSERT INTO index_metadata (id, dir_hash) VALUES (1, %s)
            ON CONFLICT (id) DO UPDATE SET dir_hash = EXCLUDED.dir_hash
        """, (current_hash,))
        self.conn.commit()
        cur.close()

    def _build_index(self):
        """Parse documents, embed them, and store in PostgreSQL."""
        cur = self.conn.cursor()

        # Clear old data
        cur.execute("TRUNCATE document_chunks RESTART IDENTITY")
        self.conn.commit()

        chunks = load_directory(self.documents_dir)
        if not chunks:
            print("[RAG] No documents found. RAG disabled.")
            cur.close()
            return

        # Embed all chunks in batches
        print(f"[RAG] Embedding {len(chunks)} chunks...")
        t0 = time.time()
        texts = [c.text for c in chunks]
        embeddings = self.embed_model.encode(texts, show_progress_bar=True, batch_size=64)
        print(f"[RAG] Embeddings computed in {time.time()-t0:.1f}s")

        # Insert into PostgreSQL
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeds = embeddings[i:i + batch_size]
            values = []
            for chunk, emb in zip(batch_chunks, batch_embeds):
                values.append((chunk.text, chunk.source_file, chunk.location, emb.tolist()))

            cur.executemany(
                "INSERT INTO document_chunks (text, source_file, location, embedding) VALUES (%s, %s, %s, %s)",
                values,
            )
            self.conn.commit()

        # Create IVFFlat index for fast search (needs at least some rows)
        chunk_count = len(chunks)
        if chunk_count >= 100:
            lists = min(chunk_count // 10, 100)
            cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
            cur.execute(f"""
                CREATE INDEX idx_chunks_embedding 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = {lists})
            """)
            self.conn.commit()

        self._save_index_meta()
        cur.close()
        print(f"[RAG] Indexed {len(chunks)} chunks into PostgreSQL + pgvector")

    def retrieve(self, query: str, top_k: int = 3) -> tuple[str, list[dict]]:
        """
        Retrieve the top-k most relevant chunks for a query using semantic search.
        Only returns chunks that meet the similarity threshold.

        Returns:
            context_str: formatted string to inject into the LLM prompt
            sources: list of {file, location, snippet, similarity} dicts for logging
        """
        t0 = time.time()

        # Embed the query
        query_embedding = self.embed_model.encode(query).tolist()

        # Query pgvector using cosine distance
        # cosine_distance = 1 - cosine_similarity, so lower = more similar
        cur = self.conn.cursor()
        cur.execute("""
            SELECT text, source_file, location, 
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))

        rows = cur.fetchall()
        cur.close()

        if not rows:
            return "", []

        sources = []
        context_parts = []

        for rank, (text, source_file, location, similarity) in enumerate(rows, 1):
            # Apply similarity threshold
            if similarity < RAG_SIMILARITY_THRESHOLD:
                print(f"[RAG]   ✗ [{similarity:.3f}] {source_file}, {location} — below threshold ({RAG_SIMILARITY_THRESHOLD})")
                continue

            context_parts.append(
                f"[Source {rank}: {source_file}, {location}]\n{text}"
            )
            snippet = text[:80] + "..." if len(text) > 80 else text
            sources.append({
                "file": source_file,
                "location": location,
                "snippet": snippet,
                "similarity": round(similarity, 3),
            })

        context_str = "\n\n".join(context_parts)
        elapsed_ms = (time.time() - t0) * 1000

        if sources:
            print(f"[RAG] Retrieved {len(sources)} chunks in {elapsed_ms:.1f}ms (query: '{query[:50]}')")
            for s in sources:
                print(f"[RAG]   → [{s['similarity']:.3f}] {s['file']}, {s['location']}: {s['snippet']}")
        else:
            print(f"[RAG] No chunks above threshold ({RAG_SIMILARITY_THRESHOLD}) for: '{query[:50]}' ({elapsed_ms:.1f}ms)")

        return context_str, sources

    def __del__(self):
        """Close database connection on cleanup."""
        if hasattr(self, 'conn') and self.conn and not self.conn.closed:
            self.conn.close()
