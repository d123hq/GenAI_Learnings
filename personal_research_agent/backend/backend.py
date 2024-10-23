# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_community.tools import TavilySearchResults
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv

_ = load_dotenv()
api_key = os.environ['GROQ_API_KEY']

llama_llm = ChatGroq(api_key=api_key,
                     model="llama3-8b-8192",
                     temperature=0.0,
                     max_retries=2)

web_search_tool = TavilySearchResults(max_results=5, search_depth="advanced", include_answer=True)

# ... [Prompt and chain definitions remain unchanged]

# Initialize FastAPI
app = FastAPI()

# Store chat history
chat_history = []

@app.post("/query/")
async def run_agent(query: QueryRequest):
    output = local_agent.invoke({"question": query.question})
    response_text = output["generation"]
    
    # Append to chat history
    chat_history.append({"user": query.question, "assistant": response_text})

    return {"generation": response_text, "chat_history": chat_history}
