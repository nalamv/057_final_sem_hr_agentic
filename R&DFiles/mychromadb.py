# chroma_vector_store.py
import os
import uuid
import numpy as np
from typing import List, Any, Dict, Optional

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

from embedding import EmbeddingPipeline


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "embeddings_collection",
        persist_dir: str = "chroma_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Chroma-backed vector store.

        - collection_name: Name of the Chroma collection.
        - persist_dir: Directory where Chroma will persist its data.
        - embedding_model: sentence-transformers model for queries.
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create a persistent Chroma client
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_dir,
            )
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.dim: Optional[int] = None  # will be inferred on first add if needed

        print(f"[INFO] Initialized Chroma collection '{self.collection_name}'")
        print(f"[INFO] Persistence directory: {self.persist_dir}")
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    # --------------------------------------------------------------------- #
    # Build from raw documents (same idea as Milvus version)
    # --------------------------------------------------------------------- #
    def build_from_documents(self, documents: List[Any]):
        """
        Chunk documents, compute embeddings, and store them in Chroma.
        Each chunk's text is stored as both document and metadata["text"].
        """
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks).astype("float32")  # (N, dim)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [{"text": t} for t in texts]

        self.add_embeddings(embeddings, texts=texts, metadatas=metadatas)
        # Persist Chroma DB to disk
        self.client.persist()
        print(
            f"[INFO] Vector store built and persisted in Chroma collection '{self.collection_name}'"
        )

    # --------------------------------------------------------------------- #
    # Add embeddings
    # --------------------------------------------------------------------- #
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        embeddings: np.ndarray shape (N, dim), dtype float32
        texts: list of strings (documents)
        metadatas: list of dicts (same length as embeddings), optional
                   if None, will create {"text": text} for each.
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D numpy array with shape (N, dim)")

        n, dim = embeddings.shape
        if len(texts) != n:
            raise ValueError("Length of texts must match number of embeddings")

        if metadatas is None:
            metadatas = [{"text": t} for t in texts]
        elif len(metadatas) != n:
            raise ValueError("Length of metadatas must match number of embeddings")

        # Track dimension for sanity (not strictly required by Chroma)
        if self.dim is None:
            self.dim = dim
        elif self.dim != dim:
            raise ValueError(
                f"Embedding dimension mismatch: store dim={self.dim}, embeddings dim={dim}"
            )

        # Generate unique IDs for each embedding
        ids = [str(uuid.uuid4()) for _ in range(n)]

        # Chroma expects Python lists for embeddings
        emb_list = embeddings.tolist()

        self.collection.add(
            ids=ids,
            embeddings=emb_list,
            documents=texts,
            metadatas=metadatas,
        )

        print(
            f"[INFO] Inserted {n} vectors into Chroma collection '{self.collection_name}'"
        )
        return ids

    # --------------------------------------------------------------------- #
    # "load" is mostly a no-op for Chroma, but we re-acquire the collection
    # --------------------------------------------------------------------- #
    def load(self):
        """
        Re-open the persistent Chroma collection (useful if you recreate the client).
        """
        self.collection = self.client.get_collection(self.collection_name)
        print(f"[INFO] Loaded Chroma collection '{self.collection_name}'")

    # --------------------------------------------------------------------- #
    # Search by embedding
    # --------------------------------------------------------------------- #
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Search Chroma for nearest neighbors.

        query_embedding: shape (1, dim) or (N, dim) - we will use only the first row.
        returns list of dicts with keys: id, distance, metadata (text & others)
        """
        if query_embedding.ndim == 1:
            vector = query_embedding.astype("float32").tolist()
        elif query_embedding.ndim == 2:
            vector = query_embedding[0].astype("float32").tolist()
        else:
            raise ValueError("query_embedding must be 1D or 2D numpy array")

        result = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["distances", "metadatas", "documents", "embeddings"],
        )

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]

        out = []
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            # ensure we always include text in metadata
            if "text" not in meta and i < len(documents):
                meta["text"] = documents[i]
            out.append(
                {
                    "id": ids[i],
                    "distance": distances[i],
                    "metadata": meta,
                }
            )

        return out

    # --------------------------------------------------------------------- #
    # Convenience: search by query text
    # --------------------------------------------------------------------- #
    def query(self, query_text: str, top_k: int = 5):
        """
        Convenience: encode query_text and search.
        """
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

    # --------------------------------------------------------------------- #
    # Drop / reset collection
    # --------------------------------------------------------------------- #
    def drop_collection(self):
        """
        Drop the collection (dangerous: removes all stored vectors & metadata).
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"[INFO] Dropped Chroma collection '{self.collection_name}'")
        except Exception as e:
            print(f"[WARN] Failed to drop collection '{self.collection_name}': {e}")


# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents

    docs = load_all_documents("data")
    chromedb=Chroma.from_documents(docs,embedding=)

    store = ChromaVectorStore(
        collection_name="demo_embeddings",
        persist_dir="chroma_store",
        embedding_model="all-MiniLM-L6-v2",
    )
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
