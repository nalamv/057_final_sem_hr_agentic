from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

print("\n--- Starting Querying from Saved DB ---")
PERSIST_DIR="./local_chroma_db"

embeddings_model=HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-l6-v2',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':False}
)

loaded_chromadb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings_model,
    collection_name="HR_Policy_data"
)

query = "How many days for Privilege Leave"
docs = loaded_chromadb.similarity_search(query)

print(f"Found relevant chunk (query: '{query}'):")
context=docs[0].page_content
print("-" * 20)
print(docs[0].page_content)
print("-" * 20)


llm_model="openai/gpt-oss-20b"
groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
print(f"[INFO] Groq LLM initialized: {llm_model}")

prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
response = llm.invoke([prompt])
print("=" * 20)
print(response.content)
print("=" * 20)