# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from companion_bot import CompanionBot

app = FastAPI()

# Instantiate bot
bot = CompanionBot(problem_phase_limit=4, wrap_up_threshold=35)

# Allow CORS for all origins (React frontend can call)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserMessage(BaseModel):
    text: str

@app.post("/message")
def send_message(msg: UserMessage):
    """
    Accepts {"text": "..."} and returns the bot response JSON:
    {
      "reply": "...",
      "mood": "😊",
      "risk": "💚",
      "stage": "companion",
      "timestamp": "..."
    }
    """
    bot_response = bot.run_once_text(msg.text)
    return bot_response

@app.get("/")
def root():
    return {"message": "Companion Bot API is running."}
