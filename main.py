import os
from langchain_community.document_loaders import PyMuPDFLoader, DocxLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_documents(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = DocxLoader(file_path)
        else:
            continue
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs