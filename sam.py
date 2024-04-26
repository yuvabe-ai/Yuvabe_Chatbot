from http.client import HTTPException
from typing import Union
from fastapi import FastAPI
from pinecone import Pinecone
import google.generativeai as genai
import os
from pydantic import BaseModel

import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "yuvabe-419516-7c5ffe0e1f9b.json"
# Initialize Pinecone index
pc = Pinecone(api_key="Pinecone-API-Key")
index = pc.Index("yuvabe")

project_id = "yuvabe-419516"
location = "us-central1"
vertexai.init(project=project_id, location=location)

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              system_instruction="yuvabe is a company located in auroville. this chatbot name is YUVA. yuvabe is an youth empowerment organization. you are a sales person for yuvabe.you act as a chatbot.yuvabe chatbot is developed by Eyuvaraj, praveen, thamaraikannan from AI Team.If the user asks a question unrelated to Yuvabe, I would say, This question doesnot pertain to Yuvabe. don't say you large language model")
chat = model.start_chat()


app = FastAPI()

genai.configure(api_key="Gemini-API-Key")


class Schema(BaseModel):
    input_text: str


def embed_text(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
        title="Embedding of single string")

    return result['embedding']


def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)


@app.post("/chat/")
async def yuva(data: Schema):
    input_text = data.input_text

    if not input_text:
        raise HTTPException(status_code=400, detail="text input is required")

    vector = embed_text(input_text)
    out = index.query(
        namespace=None,
        vector=vector,
        top_k=9,
        include_metadata=True
    )

    data = []
    for match in out.matches:
        result = match.metadata["data"]
        data.append(result)

    prompt = (f"This is the user's query:'{input_text}'. Using this data :'{data}' \
    to answer the users query. you are talking to a non-technical person, so be sure to \
    break down complicated concepts and strike a friendly and conversational tone. \
    If the passage is irrelevant to the answer, you may ignore it. \
    Return only relevant answer from question avoiding unnecessary conversation. \
    Return the response in short.")
    response = get_chat_response(chat, prompt)
    return response
