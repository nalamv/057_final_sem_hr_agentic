import chromadb
import json

print("\n--- Starting Querying from Saved DB ---")
PERSIST_DIR="./local_chroma_db"



collection_name="Employee_Tax_benefits"
client=chromadb.PersistentClient(path="./local_chroma_db")
collection=client.get_collection(name=collection_name)

print(f'collection count {collection.count()}')
print(f"--- Viewing all data in collection: '{collection_name}' ---")

all_data = collection.get(
    include=["embeddings", "documents", "metadatas"]
)

print(all_data.get("documents",[]))
print(all_data.get("ids",[]))
print(all_data.get("embeddings",[]))


