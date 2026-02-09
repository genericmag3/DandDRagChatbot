import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import re
import io
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

df = pd.read_csv("Notes.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
#add_documents = not os.path.exists(db_location)

def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_kwargs={"device": "cpu"})
    return embeddings

def parse_journal_text(file_content, databasedir):
    """Parses a text file with date headers into a structured list of dicts."""
    # Matches common date formats like 2023-10-27 or 10/27/2023 at the start of a line
    date_pattern = r'^(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})'
    
    entries = []
    current_date = "Unknown Date"
    current_content = []

    for line in file_content.splitlines():
        match = re.match(date_pattern, line.strip())
        if match:
            # If we already have a previous entry, save it before starting a new one
            if current_content:
                entries.append({
                    "Title": f"Entry for {current_date}",
                    "Date": current_date,
                    "Contents": "\n".join(current_content).strip()
                })
            current_date = match.group(1)
            current_content = [line[match.end():].strip()] # Start content after the date
        else:
            current_content.append(line.strip())

    # Catch the final entry
    if current_content:
        entries.append({
            "Title": f"Entry for {current_date}",
            "Date": current_date,
            "Contents": "\n".join(current_content).strip()
        })
    return pd.DataFrame(entries)

def create_hf_retrival_artifacts(databasedir):
    hf_embeddings = load_embeddings()
    text_splitter = SemanticChunker(hf_embeddings)
    vector_store = Chroma(
            collection_name="notes",
            persist_directory=databasedir,
            embedding_function=hf_embeddings
            )
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 9, "score_threshold": 0.1}
        )
    return text_splitter,retriever,vector_store

#def vectorize_dataframe()