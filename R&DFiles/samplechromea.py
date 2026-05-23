# hr_rag_chroma.py
import os
from pathlib import Path
from typing import List

# PDF extraction
from PyPDF2 import PdfReader
# LangChain utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # OpenAI chat model
from tqdm import tqdm

# --------------------
DATA_DIR = Path("../testdata/data")
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    text_pages = []
    for p in range(len(reader.pages)):
        page = reader.pages[p]
        try:
            txt = page.extract_text()
        except Exception:
            txt = ""
        if txt:
            # add page breaks to preserve small structure
            text_pages.append(txt + "\n\n")
    return "\n".join(text_pages)

def load_documents_from_folder(folder: Path) -> List[Document]:
    results = []
    files = list(folder.glob("*.pdf"))
    print(f"Found {len(files)} pdf(s) in {folder}")
    for f in tqdm(files, desc="Reading PDFs"):
        text = extract_text_from_pdf(f)
        # keep some metadata: filename
        results.append(Document(page_content=text, metadata={"source": str(f)}))
    return results

def chunk_documents(documents: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            md = dict(doc.metadata)
            md.update({"chunk": i})
            split_docs.append(Document(page_content=chunk, metadata=md))
    print(f"Total chunks created: {len(split_docs)}")
    return split_docs

def create_or_load_chroma(docs, embedding_model_name=EMBEDDING_MODEL, persist_directory=PERSIST_DIR):
    # embeddings via sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    # persistent write
    vectordb.persist()
    return vectordb

def build_qa_chain(vectordb):
    # create retriever
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    # Use OpenAI Chat model (requires OPENAI_API_KEY in env)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # change to gpt-4o/gpt-4 if you have access
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def main():
    # 1. load PDFs
    docs = load_documents_from_folder(DATA_DIR)
    if not docs:
        print("No PDFs found in policies_pdfs. Put your PDFs there and re-run.")
        return

    # 2. chunk
    chunks = chunk_documents(docs)

    # 3. create or load vector DB
    vectordb = create_or_load_chroma(chunks)

    # 4. make QA chain
    qa = build_qa_chain(vectordb)

    # interactive loop
    print("Ready. Ask HR policy questions (type 'exit' to quit).")
    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        resp = qa(q)
        answer = resp["result"]
        sources = resp.get("source_documents", [])
        print("\n=== Answer:\n")
        print(answer)
        print("\n--- Sources / snippets:")
        for s in sources:
            print(f"\nSource: {s.metadata.get('source')} (chunk {s.metadata.get('chunk')})")
            snippet = s.page_content.strip()
            print(snippet[:600].replace("\n", " ") + ("..." if len(snippet) > 600 else ""))

if __name__ == "__main__":
    main()
