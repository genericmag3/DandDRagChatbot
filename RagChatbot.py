from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from streamlit_lottie import st_lottie
import json
import requests
import pandas as pd
import time

import streamlit as st

# Initialize vector store
vector_store = Chroma(
    collection_name="notes",
    persist_directory="./chrome_langchain_db",
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# Set up model
model = OllamaLLM(model="phi4-mini")
model.temperature = .6

#Title streamlit chat window
st.title("D&D Q&A Chatbot 🧙‍♂️")

st.info("This app takes your notes from your campaign and passes relevant context from them along with your question to the LLM. It does not store your notes or chat history. Please consult provided references as the AI may hallucinate.")

#Grab custom spinner animation
with open("star-magic.json", "r",errors='ignore') as f:
    magic_spinner = json.load(f)

# Set up retriever in streamlit app
st.session_state.retriever = retriever

notes_uploaded = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def update_key():
    st.session_state.uploader_key += 1

note_document = None

if st.session_state.uploader_key == 0:
    placeholder = st.empty()
    # Have user upload campaign notes
    with placeholder.container():
        note_document = st.file_uploader("Upload your campaign notes", type=["csv"]) #key=st.session_state.uploader_key, on_change=update_key
else:
    notes_uploaded = True

# Create vector database from file
if note_document is not None:
    df = pd.read_csv(note_document)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db_location = "./chrome_langchain_db"
    
    if df is not None: # to do: add error checking
        documents = []
        ids = []
        
        for i, row in df.iterrows():
            document = Document(
                page_content=row["Contents"],
                metadata={"Title": row["Title"], "Date": row["Date"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)
            vector_store = Chroma(
                collection_name="notes",
                persist_directory=db_location,
                embedding_function=embeddings
                )
    if vector_store != None: # To do: add error checking
        vector_store.add_documents(documents=documents, ids=ids)

        #message_placeholder = st.empty()
        update_key()
        placeholder.empty()
        success = st.success("Campaign notes uploaded and processed successfully!")
        

    notes_uploaded = True
    



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])


#user_question = # Show chat input at the bottom when a question has been asked.
if notes_uploaded:
    user_question = st.chat_input("Ask a question about the campaign...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question,"avatar":None})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        placeholder = st.empty()
        # Display the animation initially
        with placeholder.container():
            st_lottie(magic_spinner, height=200, key="custom_spinner")

        #with st.spinner("Consulting texts..."):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful D&D adventure Q&A bot."),
            ("user", "You are an expert in answering questions about a Dungeons and Dragons campaign described in provided documents. The provided documents describe a campaign where the main protagonists are Brocc, Evryn, and Gwendolyn(Gwen). Here are the relevant documents with a date and title from the character Brocc's perspective (sometimes in first person and sometimes in third person): {notes} \n\n Here is the question to answer. Base your answer only off of the provided documents, and no other extraneous material. Do not provide references to the documents.: {question}")
        ])
            
        notes = retriever.invoke(user_question)

        chain = (
            prompt
            | model
            | StrOutputParser()
        )
        response = chain.invoke({"question": user_question, "notes": notes})  # Pass the query and relevant note documents

        response+="\n______________________________________________________\n"
        response+="Note entry References(date): \n"
        for item in notes:
            response += "* " + item.metadata["Date"] + "\n"
        response+="\n______________________________________________________\n"
        st.session_state.messages.append({"role": "assistant", "content": response, "avatar":"🧙‍♂️"})
        placeholder.empty()
        with st.chat_message("assistant", avatar="🧙‍♂️"):
            st.markdown(response)
  
