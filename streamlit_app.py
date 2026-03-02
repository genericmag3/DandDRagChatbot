from langchain_ollama import OllamaLLM
import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from streamlit_lottie import st_lottie
import json
import time
import os
import streamlit as st
import numpy as np

#import local modules
import src.utils.CreateDatabase as CreateDatabase

# Local Function definitions 

# Load LLM
@st.cache_resource
def load_model(modelname):
    model = OllamaLLM(model=modelname)
    return model

def has_subfolders(directory_path):
    if not os.path.isdir(directory_path):
        return False  # Not a valid directory

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            return True
    return False

def create_database_handler(document, databasedir, text_splitter, retriever, vector_store):
    # Start database creation
    gen = CreateDatabase.vectorize_note_document(document, databasedir, text_splitter, retriever, vector_store)

    # Grab custom spinner animation
    with open("assets/Magical_Effect_Loading.json", "r",errors='ignore') as f:
        magic_loader = json.load(f)
    animationplaceholder = st.empty()
    with animationplaceholder.container():
        st_lottie(magic_loader, height=200, key="custom_loading_spinner")
        progress_text = "Casting Vectorization Spell..."
        vectorization_progress = st.progress(0, text=progress_text)
    returnCode = None
    
     # Use a while loop to capture the return value from StopIteration
    while True:
        try:
            # Get the next yielded progress value
            progress = next(gen)
            vectorization_progress.progress(progress / 100, text= progress_text + f"{progress:.1f}%")
        except StopIteration as e:
            # This is where your 'return True' or 'return False' from the generator lives
            returnCode = e.value
            animationplaceholder.empty()
            return returnCode

def update_key():
    if st.session_state.uploader_key is not None:
        st.session_state.uploader_key += 1

# Define a generator function to stream the response word by word with a small delay to simulate typing
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

# Check for new upload
def process_sidebar_options():
    # Find local ollama models 
    local_models = ollama.list()
    local_model_names = []
    for model in local_models.models:
        local_model_names.append(model.model)

    sidebar_model_select = st.sidebar.selectbox("Select Model", local_model_names, index = None, placeholder = "Select local LLM...")
    sidebar_model_temperature = st.sidebar.selectbox("Select Model Temperature", np.round(np.linspace(0.1, 1.0, 10), 1), index = None, placeholder = "Select local LLM Temperature...")
    if ((sidebar_model_select is not None) and (sidebar_model_temperature is not None)):
        st.session_state.model_chosen = load_model(sidebar_model_select)
        st.session_state.model_chosen.temperature = sidebar_model_temperature
        
       # model = load_model(st.session_state.model_chosen)
       # model.temperature = st.session_state.model_temperature
    else:
        st.session_state.model_chosen = None
        st.session_state.model_temperature = None

def display_message_history():
    i = 0 #  represents index of references, each index can have multiple references and there is one per bot response
    for message in st.session_state.messages:  
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])
            if (message["role"] == "assistant"):
                if(st.session_state.buttoninfo[i] is not None): 
                    for buttoninfo in st.session_state.buttoninfo[i]:
                        st.button(buttoninfo[0], on_click = buttoninfo[1], args = buttoninfo[2], key = buttoninfo[3])
                i = i + 1


# Define Streamlit dialog for reference content display
@st.dialog("Reference Content")
def reference_button(content):
    st.write(content)

# Initialize session state variables for model and database upload handling
if ("uploader_key" not in st.session_state) or ("reupload_key" not in st.session_state) or ("model_chosen" not in st.session_state) or ("model_temperature" not in st.session_state):
    st.session_state.uploader_key = 0
    st.session_state.reupload_key = 0
    st.session_state.model_chosen = None 
    st.session_state.model_temperature = None
    st.session_state.notes_uploaded = False

# Ff the chat history or button info does not exist in session state, or if the user opts to re-upload notes, initialize chat history, button info, and button key in session state
if ("messages" not in st.session_state) or ("buttoninfo" not in st.session_state) or ("button_key" not in st.session_state) or (st.session_state.reupload_key == True):
    st.session_state.messages = []
    st.session_state.buttoninfo = []
    st.session_state.button_key = 0

# Initialize Streamlit UI 

st.title("TTRPG Journal Q&A Chatbot 🧙‍♂️")

st.info("This app takes your notes from your TTRPG campaign and passes your question along with relevant context from your notes to the local LLM. It does not permenantly store your notes or chat history or use them to train any model. Please consult provided references as the AI may hallucinate.")

process_sidebar_options() 

note_document = None

# Database directory location
databasedir = "database/chrome_langchain_db"

# if the database already exists, skip the upload and allow for re-upload sidebar option 
if has_subfolders(databasedir) and st.session_state.reupload_key == False:
    st.session_state.notes_uploaded = True
    update_key()
    sidebar_button = st.sidebar.button('Re-Upload Notes')
    if sidebar_button:
        if os.path.exists(databasedir):
            st.session_state.reupload_key = True
            st.rerun()

# if the database does not exist, or the user opted to re-uplaod notes, have user upload notes and create database
#elif st.session_state.uploader_key == 0 or st.session_state.reupload_key == True:
elif st.session_state.reupload_key == True:
    st.session_state.notes_uploaded = False
    placeholder = st.empty()
    # Have user upload campaign notes
    with placeholder.container():
        note_document = st.file_uploader("Upload your campaign notes")

# Init text splitter, retriever, and vector database
text_splitter, retriever, vector_store = CreateDatabase.create_hf_retrival_artifacts(databasedir)

# Check to see if user uploaded notes

completionmessage = None
# Create vector database from file if file has been uploaded by user
if note_document is not None:
    #get rid of the file uploader container once file has been selected
    placeholder.empty()
    #Clear reupload key just in case
    st.session_state.reupload_key = False
    #start data upload and database creation animation
    st.session_state.notes_uploaded = create_database_handler(note_document, databasedir, text_splitter, retriever, vector_store)
    if(st.session_state.notes_uploaded == True):
        completionmessage = st.empty()
        st.success("Journal processed successfully!")
        st.rerun()
    else:
        completionmessage = st.empty()
        st.error("Failed to vectorize database. Check file existence or disk space.")
    
display_message_history()

#Chat logic, only runs if notes have been uploaded, a model has been chosen, and a model temperature has been set
if st.session_state.notes_uploaded and (st.session_state.model_chosen is not None):
    user_question = st.chat_input("Ask a question about the campaign...")
    if user_question:
        if completionmessage is not None:
            completionmessage.empty()
        tempbuttoninfo = []
        st.session_state.messages.append({"role": "user", "content": user_question,"avatar":None})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        placeholder = st.empty()

        # Load animation from json
        with open("assets/star-magic.json", "r",errors='ignore') as f:
            magic_spinner = json.load(f)

        # Display the animation initially
        with placeholder.container():
            st_lottie(magic_spinner, height=200, key="custom_spinner")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful D&D adventure Q&A bot."),
            ("user", "You are an expert in answering questions about a Dungeons and Dragons campaign described in provided documents. "
            "The provided documents describe a campaign where the main protagonists are Brocc, Evryn, and Gwendolyn(Gwen). "
            "Here are the relevant documents with a date and title from the character Brocc's perspective (sometimes in first person and sometimes in third person): "
            "{notes} \n\n Here is the question to answer. Base your answer only off of the provided documents, and no other extraneous material. "
            "Do not provide references to the documents.: {question}")
        ])
        notes = retriever.invoke(user_question)
        chain = (
            prompt
            | st.session_state.model_chosen
            | StrOutputParser()
        )
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

                    # Add reference button information for the response to the session state            
                    st.session_state.buttoninfo.append(tempbuttoninfo) 
        # Save canned response to chat history if no references
        else:
            placeholder.empty()
            response = "Could not find any relevant journal entries for your query. It could be that there is not any relevant information regarding your query in the notes, the question needs to be reworded, or spelling needs to be reviewed."
            st.session_state.buttoninfo.append(None)
            st.write_stream(stream_data(response))
            st.session_state.messages.append({"role": "assistant", "content": response, "avatar":"🧙‍♂️"})
        st.rerun() 