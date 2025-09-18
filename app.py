# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from companion_bot import CompanionBot  # your updated bot file

app = FastAPI()
bot = CompanionBot(problem_phase_limit=4, wrap_up_threshold=35)

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)

class UserMessage(BaseModel):
    text: str

@app.post("/message")
def send_message(msg: UserMessage):
    """
    Accepts user message and returns bot reply as JSON.
    Example response:
    {
        "reply": "...",
        "mood": "😊",
        "risk": "💚",
        "stage": "companion"
    }
    """
    bot_response = bot.run_once_text(msg.text)
    return bot_response  # JSON returned directly to frontend

@app.get("/")
def root():
    return {"message": "Companion Bot API is running."}
