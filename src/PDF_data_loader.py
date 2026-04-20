from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


loader = PyPDFLoader(".././testdata/data/pdf/HolidayCalendar.pdf")
pages = loader.load()

#print(f"Content of Page 1:\n{pages[0].page_content}")
extracted_holiday_info = "\n".join([page.page_content for page in pages])
print(extracted_holiday_info)

PERSIST_DIR = "./local_chroma_db"

text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=text_splitter.create_documents([extracted_holiday_info])

embeddings_model=HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-l6-v2',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':False}
)
chunks_text=[chunk.page_content for chunk in chunks]
embeddings =embeddings_model.embed_documents(chunks_text)
print(len(embeddings))
chromadb=Chroma.from_documents(chunks,embeddings_model,persist_directory=PERSIST_DIR,collection_name="Employee_Holiday_Info")

print(f"Chroma database created and SAVED to disk at: {PERSIST_DIR}")