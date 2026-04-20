from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os

from src.constants import POLICY_PROMPT

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def do_the_similarity_search(query):
    print("--- Triggered Policy Agent  ---")
    collection_dir = "src/local_chroma_db"
    PERSIST_DIR = os.path.join(PROJECT_ROOT, collection_dir)
    print(PERSIST_DIR)
    collection_names = ["HR_Policy_data"]
    all_relevant_docs = []

    embeddings_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-l6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    for eachCollection in collection_names:
        loaded_chromadb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings_model,
            collection_name=eachCollection
        )

        loaded_chromadb.as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold': 0.7}) # Only docs with > 70% similarity
        docs = loaded_chromadb.similarity_search(query)
        print(f'Query:{query}')
        print(f'Collection: {eachCollection}: {docs}')

        if not docs:
            response = "I'm sorry, no relevant information was found in the database."
            print("No Value fetched")
        print(docs[0].page_content)
        all_relevant_docs.extend(docs)
    return all_relevant_docs

def policy_agent001(query):
    try:
        llm_model = "openai/gpt-oss-20b"
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(llm.model_name)
        context=do_the_similarity_search(query)
        prompt=POLICY_PROMPT.format(query=query,context=context)
        print(f'Input Prompt: {prompt}')
        response = llm.invoke([prompt])
        return response
    except Exception as e:
        return f"Exception Occurred in Policy Agent. {e}"

#object_name=AGENT001()
#response=policy_agent001("How many days for Privilege Leave")
#print(response.content)





