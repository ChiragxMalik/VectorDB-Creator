import os
import re
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_FOLDER = "data"
DB_FOLDER = "db"

# Load from (PDF or DOCX)
def load_document(file_path):
    if file_path.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

# Split text into chunks
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Incorporated from remote
        add_start_index=True
    )
    return splitter.split_documents(documents)

# Embedding model
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sanitize collection name
def get_clean_collection_name(file_path):
    file_name = os.path.basename(file_path)
    file_name = file_name.replace(".pdf", "").replace(".docx", "")
    clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file_name)
    clean_name = re.sub(r"^[^a-zA-Z0-9]+", "", clean_name)
    clean_name = re.sub(r"[^a-zA-Z0-9]+$", "", clean_name)
    return clean_name

# Store chunks as embeddings in ChromaDB
def store_embeddings(chunks, embedding_model, collection_name, persist_directory=DB_FOLDER):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    return vector_db

# Main pipeline
def process_all_documents(data_folder=DATA_FOLDER, db_folder=DB_FOLDER):
    embedding_model = get_embeddings_model()
    for filename in os.listdir(data_folder):
        if filename.lower().endswith(('.pdf', '.docx')):
            file_path = os.path.join(data_folder, filename)
            print(f"Processing: {file_path}")
            documents = load_document(file_path)
            chunks = split_documents(documents)
            collection_name = get_clean_collection_name(file_path)
            store_embeddings(chunks, embedding_model, collection_name, db_folder)
            print(f"Stored vectors for: {filename}\n")
#to run
if __name__ == "__main__":
    process_all_documents()
