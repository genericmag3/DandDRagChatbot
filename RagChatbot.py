from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document 
from streamlit_lottie import st_lottie
import json
import pandas as pd
import time
import os
import streamlit as st
import re
import io
from langchain.docstore.document import Document as langchaindoc
from langchain_experimental.text_splitter import SemanticChunker
from docx import Document as DocxReader

#import local modules
import CreateDatabase

# Set up model
#@st.cache_resource
def load_model(modelname):
    model = OllamaLLM(model=modelname)
    return model

#Title streamlit chat window
st.title("D&D Q&A Chatbot 🧙‍♂️")

st.info("This app takes your notes from your campaign and passes relevant context from them along with your question to the LLM. It does not store your notes or chat history. Please consult provided references as the AI may hallucinate.")

model = load_model("phi4:14b")
model.temperature = .7

#Grab custom spinner animation
with open("star-magic.json", "r",errors='ignore') as f:
    magic_spinner = json.load(f)

#Grab custom file upload animation
with open("Magical_Effect_Loading.json", "r",errors='ignore') as f:
    magic_loader = json.load(f)

notes_uploaded = False

if "first_chat_key" not in st.session_state:
    st.session_state.first_chat_key = 0

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def update_key():
    st.session_state.uploader_key += 1

note_document = None

def has_subfolders(directory_path):
    if not os.path.isdir(directory_path):
        return False  # Not a valid directory

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            return True
    return False

databasedir = "./chrome_langchain_db"

# if the database already exists, skip the upload. 
# To do: allow for re-upload of notes via sidebar button
if has_subfolders(databasedir):
    notes_uploaded = True
    update_key()
elif st.session_state.uploader_key == 0:
    placeholder = st.empty()
    # Have user upload campaign notes
    with placeholder.container():
        note_document = st.file_uploader("Upload your campaign notes") #key=st.session_state.uploader_key, on_change=update_key

# Init text splitter, retriever, and vector database
text_splitter, retriever,vector_store = CreateDatabase.create_hf_retrival_artifacts(databasedir)

success = None
# Create vector database from file if file has been uploaded
if note_document is not None:
    #get rid of the file uploader container once file has been selected
    placeholder.empty()
    #start data upload and database creation animation
    animationplaceholder = st.empty()
        # Display the animation initially
    with animationplaceholder.container():
        st_lottie(magic_loader, height=200, key="custom_loading_spinner")
        progress_text = "Casting Vectorization Spell..."
        vectorization_progress = st.progress(0, text=progress_text)

    file_extension = note_document.name.split('.')[-1].lower()

    if file_extension == 'csv':
        df = pd.read_csv(note_document)
    elif file_extension == 'docx':
        # Read the file into a buffer
        bytes_data = note_document.read()
        doc_io = io.BytesIO(bytes_data)
        document = DocxReader(doc_io)

        document_text = []
        for paragraph in document.paragraphs:
            document_text.append(paragraph.text)
            
        # Join paragraphs together with newline character
        text_content = '\n'.join(document_text)

        df = CreateDatabase.parse_journal_text(text_content, databasedir)
    else:
        # Read the text file content and parse it into the same dataframe structure
        stringio = io.StringIO(note_document.getvalue().decode("utf-8"))
        df = CreateDatabase.parse_journal_text(stringio.read(), databasedir)

    if df is not None and not df.empty:
        documents = []
        idlist = []
        l = 0 
        
        for i, row in df.iterrows():
            text = str(row["Contents"])
            # Existing semantic chunking via text_splitter
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                document = langchaindoc(
                    page_content=chunk,
                    metadata={
                        "Title": row.get("Title", "Untitled"), 
                        "Date": str(row.get("Date", "Unknown")), 
                        "Exerpt Start": chunk[:25], 
                        "Exerpt End": chunk[-25:]
                    },
                    id=str(l)
                )
                idlist.append(str(l))
                documents.append(document)
                l += 1
            
            # Progress bar logic
            percent_complete = (i + 1) / len(df) * 100
            vectorization_progress.progress(int(percent_complete), text=progress_text)
                
        if vector_store is not None:
            vector_store.add_documents(documents=documents, ids=idlist)
        
        animationplaceholder.empty()
        notes_uploaded = True
        st.success("Journal processed successfully!")
   

# Initialize session state variables
if ("messages" not in st.session_state) or ("buttoninfo" not in st.session_state) or ("button_key" not in st.session_state):
    st.session_state.messages = []
    st.session_state.buttoninfo = []
    st.session_state.button_key = 0


i = 0 #  represents index of references, each index can have multiple references and there is one per bot response
# Display chat messages and references from history on app rerun
for message in st.session_state.messages:  
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])
        if (message["role"] == "assistant"):
            if(st.session_state.buttoninfo[i] is not None): 
                for buttoninfo in st.session_state.buttoninfo[i]:
                    st.button(buttoninfo[0], on_click = buttoninfo[1], args = buttoninfo[2], key = buttoninfo[3])
            i = i + 1

@st.dialog("Reference Content")
def reference_button(content):
    st.write(content)

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

if notes_uploaded:
    user_question = st.chat_input("Ask a question about the campaign...")
    if user_question:
        if success is not None:
            success.empty()
        tempbuttoninfo = []
        st.session_state.messages.append({"role": "user", "content": user_question,"avatar":None})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        placeholder = st.empty()
        # Display the animation initially
        with placeholder.container():
            st_lottie(magic_spinner, height=200, key="custom_spinner")

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
        #references_found = False
        if len(notes) > 0:
            response = chain.invoke({"question": user_question, "notes": notes})  # Pass the query and relevant note documents
            placeholder.empty()
            references_found = True
            with st.chat_message("assistant", avatar="🧙‍♂️"):

            # Only display references if any were found
                if(references_found):
                    response +="\n______________________________________________________\n"
                    response += "Note entry References: \n"
                    st.session_state.messages.append({"role": "assistant", "content": response, "avatar":"🧙‍♂️"})
                    st.write_stream(stream_data(response))
                    # Create a unique button for each reference
                    for item in notes:
                        tempbuttoninfo.append([item.metadata["Date"],reference_button, (item.page_content,), f"click_{st.session_state.button_key}"])
                        st.button(str(item.metadata["Date"]), on_click= reference_button,args=(item.page_content,),  key = f"click_{st.session_state.button_key}")
                        time.sleep(0.02)
                        # Generate new button key for next button
                        st.session_state.button_key = st.session_state.button_key + 1

                    # Add button information to the session state            
                    st.session_state.buttoninfo.append(tempbuttoninfo)
        # Save canned response to chat history if no references
        else:
            placeholder.empty()
            response = "Could not find any relevant journal entries for your query. It could be that there is not any relevant information regarding your query in the notes, the question needs to be reworded, or spelling needs to be reviewed."
            st.session_state.buttoninfo.append(None)
            st.write_stream(stream_data(response))
            st.session_state.messages.append({"role": "assistant", "content": response, "avatar":"🧙‍♂️"})
        st.rerun()
        