import streamlit as st  # Importing Streamlit for the web interface
from PyPDF2 import PdfReader  # Importing PyPDF2 for reading PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing Langchain's text splitter
from langchain_core.prompts import ChatPromptTemplate  # Importing ChatPromptTemplate from Langchain
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings  # Importing SpacyEmbeddings
from langchain_community.vectorstores import FAISS  # Importing FAISS for vector store
from langchain.tools.retriever import create_retriever_tool  # Importing retriever tool from Langchain
from dotenv import load_dotenv  # Importing dotenv to manage environment variables
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Importing ChatOpenAI and OpenAIEmbeddings from Langchain
from langchain.agents import AgentExecutor, create_tool_calling_agent  # Importing agent-related modules from Langchain
from dotenv import load_dotenv, find_dotenv
import os
import openai
load_dotenv(find_dotenv())


openai.api_key = os.environ['OPENAI_API_KEY']
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text +=page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                                   chunk_overlap = 200,
                                                   )
    chunks = text_splitter.text(text)
    return chunks

embeddings = OpenAIEmbeddings()

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_db")
    return vector_store

def get_conversational_chain(llm, tools, query):
    
    prompt = ChatPromptTemplate.from_messages(
        [
            
            ("system", "You are a helpful assistant. Answer questions truthfully and in detail as possible from the context. If the contenxt is not provided, answer 'Context is unavailable'"),
            ("placeholder", "{chat_history}"),
            ("human","input"),
            ("placeholder", "{agent_scratchpad}"),
            
        ]
    )
        # Create the tool list and the agent
    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)
    
    # Execute the agent to process the user's query and get a response
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response = agent_executor.invoke({"input": ques})
    return(response['output'])

llm = ChatOpenAI(api_key = openai.api_key,
                       model = 'gpt-4o-mini',
                       temperature = 0)

def user_input(user_question):
    # Load the vector database
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    
    # Create a retriever from the vector database
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answers to queries from the PDF")
    
    # Get the conversational chain to generate a response
    get_conversational_chain(llm, retrieval_chain, user_question)
    
def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
