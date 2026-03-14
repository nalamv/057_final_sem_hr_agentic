from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os

from src.constants import POLICY_PROMPT


def do_the_similarity_search(query):
    print("--- Triggered Holiday Agent  ---")
    PERSIST_DIR = "C:/Users/yugan/Desktop/VNIT Mtech/GDrive/SEM-3/NLP/FinalProject/src/local_chroma_db"
    #print(PERSIST_DIR)
    collection_names = "Employee_Holiday_Info"
    all_relevant_docs = []

    embeddings_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-l6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    loaded_chromadb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings_model,
        collection_name=collection_names
    )

    loaded_chromadb.as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold': 0.7}) # Only docs with > 70% similarity
    docs = loaded_chromadb.similarity_search(query)

    if not docs:
        response = "I'm sorry, no relevant information was found in the database."
    #print(docs[0].page_content)
    all_relevant_docs.extend(docs)
    return all_relevant_docs

def holiday_agent001(query):
    try:
        llm_model = "openai/gpt-oss-20b"
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        context=do_the_similarity_search(query)
        prompt=POLICY_PROMPT.format(query=query,context=context)
        response = llm.invoke([prompt])
        return response
    except Exception as e:
        print(e)
        return f"Exception Occurred in Policy Agent. {e}"

#object_name=AGENT001()
#response=policy_agent001("How many days for Privilege Leave")
#print(response.content)





