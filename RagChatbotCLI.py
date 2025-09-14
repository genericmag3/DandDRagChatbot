from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from termcolor import colored
import emoji as moji


vector_store = Chroma(
    collection_name="notes",
    persist_directory="./chrome_langchain_db",
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)

model = OllamaLLM(model="phi4:14b")
model.temperature = .6

template = """
You are an expert in answering questions about a Dungeons and Dragons campaign described in provided documents. The provided documents describe a campaign where the main protaganists are Brocc, Evryn, and Gwendolyn(Gwen).

Here are the relevant documents with a date and title from the character Brocc's perspective (sometimes in first person and sometimes in third person): {notes}
    
Here is the question to answer. Base your answer only off of the provided documents, and no other extraineous material: {question}
"""
prompt = ChatPromptTemplate.from_template(template, model=model)


chain = prompt | model 


from datetime import datetime

print(moji.emojize(colored("\n This is your D&D adventure Q&A bot :man_mage:. I take strucutred D&D notes as input and output answers to questions with references to specific sesssion note entries. This bot is a work in progress so make sure to read answers carfully and check note references for accuracy. Input 'q' to exit chatbot.\n", "green")))

while True:
    current_date = datetime.now()
    user_input = input(colored("Ask a question about the campaign: ", "green"))

    if user_input.lower() == "q":
        print(colored("\n I hope I proved useful. Don't forget to 'get his ass'.\n", "green"))
        break

    # Perform a similarity search in the vector database
    notes = retriever.invoke(user_input)
    
    # Generate a response using the language model
    resp = chain.invoke(
        {
            "question": user_input,
            "notes": notes,
            "current_date": current_date,
        }
    )
    #print("\n______________________________________________________\n")
    
        
    print("\n______________________________________________________\n")
    print(colored("\nD&D Campaign Q&A Bot: ", "green") + colored(resp, "blue"))
    print("\n______________________________________________________\n")
    print(colored("Note entry References(Title, date): \n", "green"))
    for item in notes:
        print("* " + colored(item.metadata["title"] + "," + item.metadata["date"], "yellow") + "\n")

    print("______________________________________________________\n")