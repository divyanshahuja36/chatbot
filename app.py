from fastapi import FastAPI
from pydantic import BaseModel
from chatbot.companion_bot import CompanionBot

app = FastAPI()
bot = CompanionBot(checkin_interval_seconds=60*60*4, problem_phase_limit=4, wrap_up_threshold=35)

class UserMessage(BaseModel):
    text: str

@app.post("/message")
def send_message(msg: UserMessage):
    # Returns bot reply for frontend
    reply = bot.run_once_text(msg.text)
    # run_once_text already handles display internally; we may modify to return string:
    return {"reply": "Message processed. Check logs for detailed reply."}

@app.get("/")
def root():
    return {"message": "Companion Bot API is running."}
