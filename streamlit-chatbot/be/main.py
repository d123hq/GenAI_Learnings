import os
import openai
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

llm_openai = ChatOpenAI(
    api_key = openai.api_key,
    model = 'gpt-4o-mini',
    temperature = 0
)

# response = llm.invoke('What is the capital of France?')
# print(response)

# Initialize FastAPI
app = FastAPI()

# Store chat history
chat_history = []

class QueryRequest(BaseModel):
    question: str

@app.post("/query/")
async def answer_query(query: QueryRequest):
    output = llm.invoke({"question": query.question})
    response_text = output
    
    # Append to chat history
    chat_history.append({"user": query.question, "assistant": response_text})

    return {"generation": response_text, "chat_history": chat_history}





