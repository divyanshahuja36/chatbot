from fastapi import FastAPI
from pydantic import BaseModel
from companion_bot import CompanionBot  # updated import

app = FastAPI()
# Only pass valid arguments
bot = CompanionBot(problem_phase_limit=4, wrap_up_threshold=35)

class UserMessage(BaseModel):
    text: str

@app.post("/message")
def send_message(msg: UserMessage):
    # Returns bot reply for frontend
    bot.run_once_text(msg.text)
    return {"reply": "Message processed. Check logs for detailed reply."}

@app.get("/")
def root():
    return {"message": "Companion Bot API is running."}
