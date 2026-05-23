import tika
from tika import parser
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


#start tika sever http:localhost:9998
tika_server='http://localhost:9998'
file_pdf= "../testdata/data/Leave-and-Holiday-Policy.pdf"
PERSIST_DIR = "./local_chroma_db"

text=parser.from_file(file_pdf,tika_server)
print(text["content"])

text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=text_splitter.create_documents([text["content"]])
embeddings_model=HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-l6-v2',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':False}
)
chunks_text=[chunk.page_content for chunk in chunks]
embeddings =embeddings_model.embed_documents(chunks_text)
print(len(embeddings))
chromadb=Chroma.from_documents(chunks,embeddings_model,persist_directory=PERSIST_DIR,collection_name="HR_Policy_data")

print(f"Chroma database created and SAVED to disk at: {PERSIST_DIR}")
print("\n--- Starting Querying from Saved DB ---")

query="How many days for Privilege Leave"

docs=chromadb.similarity_search(query)

print(docs[0].page_content)



