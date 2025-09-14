import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings

df = pd.read_csv("Notes.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)