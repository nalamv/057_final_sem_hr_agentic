# milvus_vector_store.py
import numpy as np
from typing import List, Any, Dict
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from embedding import EmbeddingPipeline

class MilvusVectorStore:
    def __init__(
        self,
        collection_name: str = "embeddings_collection",
        host: str = "127.0.0.1",
        port: str = "19530",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        dim: int | None = None,
        index_params: Dict = None
    ):
        """
        Milvus-backed vector store.

        - collection_name: Milvus collection where vectors are stored.
        - host/port: Milvus server address.
        - embedding_model: sentence-transformers model for queries (kept for compatibility).
        - dim: vector dimension. If None, will be inferred at first insert.
        - index_params: dict passed to create_index (defaults to IVF_FLAT parameters).
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dim = dim
        self.collection: Collection | None = None
        self.index_params = index_params or {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }

        # Connect to Milvus
        connections.connect(host=self.host, port=self.port)
        print(f"[INFO] Connected to Milvus at {self.host}:{self.port}")
        # If collection exists, get it, otherwise it will be created at first add_embeddings
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            print(f"[INFO] Loaded existing collection: {self.collection_name}")
        else:
            self.collection = None
            print(f"[INFO] Collection {self.collection_name} does not exist yet.")

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def _create_collection(self, dim: int):
        """
        Create a Milvus collection with a float vector field and a text field for metadata.
        Uses an auto-generated int64 primary key.
        """
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return

        # Fields: pk (primary), emb (float vector), text (string)
        pk = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True)
        emb = FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=dim)
        text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)

        schema = CollectionSchema(fields=[pk, emb, text], description="Embeddings collection")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"[INFO] Created collection {self.collection_name} with dim={dim}")

        # Create index on the vector field
        self.collection.create_index(field_name="emb", index_params=self.index_params)
        # Load collection into memory for search
        self.collection.load()
        print(f"[INFO] Created index on 'emb' and loaded collection into memory")

    def build_from_documents(self, documents: List[Any]):
        """
        Chunk documents, compute embeddings, store to Milvus along with metadata (text).
        """
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks).astype('float32')  # (N, dim)
        metadatas = [chunk.page_content for chunk in chunks]
        self.add_embeddings(embeddings, metadatas)
        print(f"[INFO] Vector store built and persisted in Milvus collection '{self.collection_name}'")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[str] = None):
        """
        embeddings: np.ndarray shape (N, dim), dtype float32
        metadatas: list of strings (same length as embeddings)
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D numpy array with shape (N, dim)")

        n, dim = embeddings.shape
        if self.collection is None:
            # create collection with detected dim
            self._create_collection(dim)
            self.dim = dim
        else:
            # ensure dims match
            if self.dim is None:
                self.dim = dim
            elif self.dim != dim:
                raise ValueError(f"Embedding dimension mismatch: collection dim={self.dim}, embeddings dim={dim}")

        # Milvus expects a columnar list of lists for insert
        emb_list = embeddings.tolist()
        # metadata (text) must align; if None, store empty strings
        if metadatas is None:
            metadatas = ["" for _ in range(n)]
        elif len(metadatas) != n:
            raise ValueError("Length of metadatas must match number of embeddings")

        # Insert: columns order must follow schema: [emb, text] (pk is auto)
        # For pymilvus, pass data as list-of-columns: [[emb_vecs], [text_values]]
        insert_result = self.collection.insert([emb_list, metadatas])
        # After insertion, optionally flush to persist
        self.collection.flush()
        # Optionally, you can get ids (auto-generated) using insert_result.primary_keys
        print(f"[INFO] Inserted {n} vectors into Milvus collection '{self.collection_name}'")
        return insert_result.primary_keys

    def load(self):
        """
        Ensure collection is available and loaded into memory.
        """
        if not utility.has_collection(self.collection_name):
            raise RuntimeError(f"Collection {self.collection_name} does not exist in Milvus")
        self.collection = Collection(self.collection_name)
        self.collection.load()
        # update dim if unknown
        # dim can be read from schema
        for field in self.collection.schema.fields:
            if field.name == "emb":
                self.dim = field.params.get("dim", self.dim)
        print(f"[INFO] Loaded collection '{self.collection_name}' into memory. dim={self.dim}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5, params: Dict = None):
        """
        Search Milvus for nearest neighbors.
        query_embedding: shape (1, dim) or (N, dim)
        returns list of dicts with keys: index (milvus id), distance, metadata (text)
        """
        if self.collection is None:
            raise RuntimeError("Collection not initialized or loaded. Call load() first or add embeddings.")

        if query_embedding.ndim == 1:
            vectors = [query_embedding.tolist()]
        elif query_embedding.ndim == 2:
            vectors = query_embedding.tolist()
        else:
            raise ValueError("query_embedding must be 1D or 2D numpy array")

        search_params = params or {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.HRPolicies(
            data=vectors,
            anns_field="emb",
            param=search_params,
            limit=top_k,
            output_fields=["text"]  # request the text metadata back
        )

        # results is a list (for each query) of list of Hit objects
        out = []
        for hits in results:
            row = []
            for hit in hits:
                # hit.id is the primary key, hit.distance is distance (L2)
                row.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "metadata": {"text": hit.entity.get("text")} if hit.entity else None
                })
            out.append(row)
        # If single query, return first list
        return out[0] if len(out) == 1 else out

    def query(self, query_text: str, top_k: int = 5):
        """
        Convenience: encode text and search
        """
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

    def drop_collection(self):
        """Drop the collection (dangerous)."""
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()
            print(f"[INFO] Dropped collection {self.collection_name}")

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = MilvusVectorStore(collection_name="demo_embeddings", host="localhost", port="19530")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
