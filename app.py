from fastapi import FastAPI

from bot import ask_chatbot

app = FastAPI()

@app.get("/chat")
async def chat(question: str = ""):
    response = ask_chatbot(question)
    return response