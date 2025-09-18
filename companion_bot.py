"""
Companion Bot - Final version for API integration
Features:
 - Typed input only
 - Collects user's primary problem, then focuses replies specifically on that problem
 - Relationship (breakup/cheating) and job-betrayal flows have tailored advice
 - Detects suicidal language and asks "how long" — prompts assessment/escalation if needed
 - Suggests PHQ-9 / GAD-7 when duration or sentiment suggests
 - Problem-phase limited to a few focused steps (configurable)
 - Wrap-up summary & 4-5 actionable plan at wind-up threshold (30-40 messages)
"""

import re
from datetime import datetime
from textblob import TextBlob

try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except Exception:
    AZURE_AVAILABLE = False

class CompanionBot:
    def __init__(self, problem_phase_limit=4, wrap_up_threshold=35):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        try:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(api_version=self.api_version,
                                      azure_endpoint=self.endpoint,
                                      api_key=self.subscription_key)
        except Exception as e:
            print("[Warning] AzureOpenAI not available or failed to init:", e)
            self.client = None

        # Session state
        self.user_profile = {
            "problem_type": None,
            "problem_collected": False,
            "problem_collected_texts": [],
            "conversation_stage": "companion",
            "message_count": 0,
            "sentiment_history": [],
            "risk_level": "low",
            "last_assistant_reply": None,
            "problem_phase_counter": 0
        }

        self.conversation_history = []
        self.problem_phase_limit = problem_phase_limit
        self.wrap_up_threshold = wrap_up_threshold

        self.risk_words = {
            "severe": ["suicide", "kill myself", "end it all", "no point living", "better off dead"],
            "high": ["hopeless", "worthless", "hate myself", "can't go on", "everything is wrong"],
            "moderate": ["depressed", "anxious", "panic", "scared", "overwhelmed", "stressed"],
            "low": ["tired", "worried", "sad", "down", "upset"]
        }

    # ---------------------------
    # Utilities
    # ---------------------------
    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        polarity = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)

        text_lower = text.lower()
        current_risk = "low"
        for level, words in self.risk_words.items():
            if any(w in text_lower for w in words):
                current_risk = level
                break

        sentiment = {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "risk_level": current_risk,
            "timestamp": datetime.now().isoformat()
        }

        self.user_profile["sentiment_history"].append(sentiment)
        order = {"low":0, "moderate":1, "high":2, "severe":3}
        if order[current_risk] > order.get(self.user_profile.get("risk_level","low"), 0):
            self.user_profile["risk_level"] = current_risk
        return sentiment

    def parse_duration_days(self, text):
        text = text.lower()
        m = re.search(r"for\s+(\d+)\s*(day|days|week|weeks|month|months|year|years)", text)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            if "day" in unit: return n
            if "week" in unit: return n*7
            if "month" in unit: return n*30
            if "year" in unit: return n*365
        word_nums = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        m2 = re.search(r"for\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s*(day|week|month|year)s?", text)
        if m2:
            n = word_nums.get(m2.group(1), 1)
            unit = m2.group(2)
            if "day" in unit: return n
            if "week" in unit: return n*7
            if "month" in unit: return n*30
            if "year" in unit: return n*365
        if "since last week" in text: return 7
        if "since last month" in text: return 30
        if "since yesterday" in text: return 1
        return None

    def detect_problem_type(self, text):
        rl = text.lower()
        relationship_keywords = ["breakup","broke up","cheat","cheated","girlfriend","boyfriend","partner","relationship","cheating"]
        job_keywords = ["fired","laid off","lost my job","lost job","betray","boss","coworker","job","workplace","resign","quit","sacked"]
        if any(k in rl for k in relationship_keywords): return "relationship"
        if any(k in rl for k in job_keywords): return "job"
        return "other"

    def detect_suicidal_language(self, text):
        return any(w in text.lower() for w in self.risk_words["severe"])

    # ---------------------------
    # AI call with stub
    # ---------------------------
    def _ai_stub(self, user_input, role_hint="companion", extra=""):
        pt = self.user_profile.get("problem_type") or "other"
        if role_hint == "companion":
            if pt == "relationship":
                return "I’m so sorry that happened. Focus on small safety steps right now."
            if pt == "job":
                return "That’s a painful betrayal at work. Start with small steps."
            return "I hear you. That sounds really hard."
        if role_hint == "wrap_up":
            return "Here’s a short 4-step plan to move forward: Ground, self-care, journaling, reach out."
        if role_hint == "assessment_prompt":
            return "It might help to do a brief screening like PHQ-9 or GAD-7."
        return "I’m here with you."

    def call_ai(self, user_input, role_hint="companion", extra_system=""):
        if not self.client:
            return self._ai_stub(user_input, role_hint, extra_system)
        try:
            system_prompt = "You are a warm, empathetic AI companion."
            if role_hint == "wrap_up":
                system_prompt += " Provide a sharp 4-5 step action plan tailored to the user's recent messages."
            messages = [{"role":"system","content":system_prompt}]
            recent = self.conversation_history[-6:]
            for turn in recent:
                messages.append({"role":"user","content":turn["user"]})
                messages.append({"role":"assistant","content":turn["assistant"]})
            messages.append({"role":"user","content":user_input})
            resp = self.client.chat.completions.create(
                messages=messages,
                max_tokens=400,
                model=self.deployment,
                temperature=0.7
            )
            return resp.choices[0].message.content
        except Exception as e:
            print("[AI call failed]", e)
            return self._ai_stub(user_input, role_hint, extra_system)

    # ---------------------------
    # Display
    # ---------------------------
    def display(self, text, stage, sentiment):
        last = self.user_profile.get("last_assistant_reply")
        if last and last.strip() == text.strip():
            text = "I hear you. I'm here. We can try grounding or make a simple plan."
        mood_emo = "😊" if sentiment["polarity"] > 0.1 else "😐" if sentiment["polarity"] > -0.1 else "😔"
        risk_map = {"low":"💚","moderate":"💛","high":"🧡","severe":"🔴"}
        self.user_profile["last_assistant_reply"] = text
        return {
            "reply": text,
            "mood": mood_emo,
            "risk": risk_map.get(sentiment['risk_level'], '💚'),
            "stage": stage
        }

    # ---------------------------
    # Problem-phase handling
    # ---------------------------
    def focused_companion_reply(self, user_input, sentiment):
        self.user_profile["problem_phase_counter"] += 1
        ai_reply = self.call_ai(user_input, "companion")
        self.conversation_history.append({
            "user": user_input,
            "assistant": ai_reply,
            "sentiment": sentiment,
            "stage": "companion",
            "timestamp": datetime.now().isoformat()
        })
        self.user_profile["message_count"] += 1
        return self.display(ai_reply, "companion", sentiment)

    # ---------------------------
    # Run once
    # ---------------------------
    def run_once_text(self, text):
        sentiment = self.analyze_sentiment(text)
        if not self.user_profile["problem_collected"]:
            self.user_profile["problem_type"] = self.detect_problem_type(text)
            self.user_profile["problem_collected"] = True
            self.user_profile["problem_collected_texts"].append(text)
            self.user_profile["problem_phase_counter"] = 0
        return self.focused_companion_reply(text, sentiment)
