"""
Companion Bot - Final version focused problem collector & sharp wrap-up
install these pip install openai TextBlob
Features:
 - Typed input only
 - Collects user's primary problem, then focuses replies specifically on that problem
 - Relationship (breakup/cheating) and job-betrayal flows have tailored advice
 - Detects suicidal language and asks "how long" — prompts assessment/escalation if needed
 - Suggests PHQ-9 / GAD-7 when duration or sentiment suggests
 - Problem-phase limited to a few focused steps (configurable)
 - Wrap-up summary & 4-5 actionable plan at wind-up threshold (30-40 messages)
 - Minimal hardcoded keys per user instruction (replace with env vars if desired)
"""

import re
from datetime import datetime

from textblob import TextBlob

# Try to import AzureOpenAI; if not available, code will use stubbed responses.
try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except Exception:
    AZURE_AVAILABLE = False


class CompanionBot:
    def __init__(self,
                 problem_phase_limit=4,
                 wrap_up_threshold=35):
        # Load environment variables
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        # Get Azure OpenAI configuration from environment variables
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        # Init client if available
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
            "age": None,
            "occupation": None,
            "main_concerns": [],
            "problem_type": None,  # 'relationship', 'job', 'other'
            "problem_collected": False,
            "problem_collected_texts": [],
            "conversation_stage": "companion",
            "message_count": 0,
            "sentiment_history": [],
            "risk_level": "low",
            "last_assistant_reply": None,
            "duration_flagged": False,
            "problem_phase_counter": 0
        }

        self.conversation_history = []
        self.problem_phase_limit = problem_phase_limit
        self.wrap_up_threshold = wrap_up_threshold

        # Detection maps
        self.risk_words = {
            "severe": ["suicide", "kill myself", "end it all", "no point living", "better off dead"],
            "high": ["hopeless", "worthless", "hate myself", "can't go on", "everything is wrong"],
            "moderate": ["depressed", "anxious", "panic", "scared", "overwhelmed", "stressed"],
            "low": ["tired", "worried", "sad", "down", "upset"]
        }

    # ---------------------------
    # Utilities: sentiment, duration, detection
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
        # escalate stored risk
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
        # word numbers
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
        relationship_keywords = ["breakup", "broke up", "cheat", "cheated", "girlfriend", "boyfriend", "partner", "relationship", "cheating"]
        job_keywords = ["fired", "laid off", "lost my job", "lost job", "betray", "boss", "coworker", "job", "workplace", "resign", "quit", "sacked"]
        if any(k in rl for k in relationship_keywords): return "relationship"
        if any(k in rl for k in job_keywords): return "job"
        return "other"

    def detect_suicidal_language(self, text):
        tl = text.lower()
        for w in self.risk_words["severe"]:
            if w in tl:
                return True
        return False

    # ---------------------------
    # LLM call (Azure) with stub fallback
    # ---------------------------
    def _ai_stub(self, user_input, role_hint="companion", extra=""):
        pt = self.user_profile.get("problem_type") or "other"
        if role_hint == "companion":
            if pt == "relationship":
                return ("I’m so sorry that happened. Focus on small safety steps right now: breathe slowly, sip water, and let yourself feel what comes.")
            if pt == "job":
                return ("That’s a painful betrayal at work. Start with small steps: take a break, breathe, and write down what happened in one short paragraph.")
            return ("I hear you. That sounds really hard. I can suggest grounding exercises or a short plan to get through the next few hours.")
        if role_hint == "wrap_up":
            return ("Here’s a short 4-step plan to move forward: 1) Ground & stabilize now 2) Short self-care 3) Journaling or assessment 4) Reach out / schedule follow-up.")
        if role_hint == "assessment_prompt":
            return ("It might help to do a brief screening like PHQ-9 (depression) or GAD-7 (anxiety). Would you like to try one now?")
        return "I’m here with you."

    def call_ai(self, user_input, role_hint="companion", extra_system=""):
        if not self.client:
            return self._ai_stub(user_input, role_hint, extra_system)
        try:
            system_prompt = "You are a warm, empathetic AI companion. Be concise and supportive."
            if role_hint == "wrap_up":
                system_prompt += " Provide a sharp 4-5 step action plan tailored to the user's recent messages."
            messages = [{"role":"system", "content": system_prompt}]
            recent = self.conversation_history[-6:]
            for turn in recent:
                messages.append({"role":"user", "content": turn["user"]})
                messages.append({"role":"assistant", "content": turn["assistant"]})
            messages.append({"role":"user", "content": user_input})
            resp = self.client.chat.completions.create(messages=messages,
                                                      max_tokens=400,
                                                      model=self.deployment,
                                                      temperature=0.7)
            return resp.choices[0].message.content
        except Exception as e:
            print("[AI call failed]", e)
            return self._ai_stub(user_input, role_hint, extra_system)

    # ---------------------------
    # Problem-phase: collect + focused replies (bounded)
    # ---------------------------
    def focused_companion_reply(self, user_input, sentiment):
        self.user_profile["problem_phase_counter"] += 1
        ptype = self.user_profile.get("problem_type", "other")
        if self.detect_suicidal_language(user_input):
            print("\n⚠️ You mentioned suicidal thoughts.")
            duration = input("How long have you felt like this? (e.g., 'a few days', '2 weeks', 'months'): ").strip()
            days = self.parse_duration_days(duration) or None
            if days and days >= 14:
                print("Consider a PHQ-9 / GAD-7 and immediate support.")
            print("If you are in immediate danger, please call emergency services now.")
            self.escalate_to_human()
            return

        ai_reply = self.call_ai(user_input, "companion")
        self.conversation_history.append({"user": user_input, "assistant": ai_reply, "sentiment": sentiment, "stage":"companion", "timestamp": datetime.now().isoformat()})
        self.user_profile["message_count"] += 1
        self.display(ai_reply, "companion", sentiment)

        if self.user_profile["problem_phase_counter"] >= self.problem_phase_limit:
            self.create_action_plan_and_wrap()

    # ---------------------------
    # Display helper
    # ---------------------------
    def display(self, text, stage, sentiment):
        last = self.user_profile.get("last_assistant_reply")
        if last and last.strip() == text.strip():
            text = "I hear you. I'm here. We can try grounding or make a simple plan."
        print("\n" + "="*60)
        print(f"🤖 Companion ({stage.title()}):")
        print(text)
        mood_emo = "😊" if sentiment["polarity"] > 0.1 else "😐" if sentiment["polarity"] > -0.1 else "😔"
        risk_map = {"low":"💚","moderate":"💛","high":"🧡","severe":"🔴"}
        print(f"\n📊 Mood: {mood_emo} | Risk: {risk_map.get(sentiment['risk_level'],'💚')}")
        print("="*60)
        self.user_profile["last_assistant_reply"] = text

    # ---------------------------
    # Main pipeline for user input
    # ---------------------------
    def handle_user_input(self, user_input):
        sentiment = self.analyze_sentiment(user_input)

        if self.detect_suicidal_language(user_input):
            print("\n⚠️ You used language that suggests severe distress.")
            dur = input("Duration (e.g., 'a few days', '2 weeks'): ").strip()
            days = self.parse_duration_days(dur) or None
            if days and days >= 14:
                print("Recommend taking a short assessment (PHQ-9 / GAD-7) and reaching out to a clinician.")
            print("If you feel you might act on these thoughts, contact emergency services immediately.")
            self.escalate_to_human()
            self.conversation_history.append({"user":user_input,"assistant":"[crisis_handled]","sentiment":sentiment,"stage":"crisis","timestamp":datetime.now().isoformat()})
            return

        if not self.user_profile["problem_collected"]:
            ptype = self.detect_problem_type(user_input)
            self.user_profile["problem_type"] = ptype
            self.user_profile["problem_collected"] = True
            self.user_profile["problem_collected_texts"].append(user_input)
            d = self.parse_duration_days(user_input)
            if d and d >= 14:
                self.user_profile["duration_flagged"] = True
            self.user_profile["problem_phase_counter"] = 0
            self.focused_companion_reply(user_input, sentiment)
            return

        if self.user_profile["problem_phase_counter"] < self.problem_phase_limit:
            self.focused_companion_reply(user_input, sentiment)
            return
        else:
            self.create_action_plan_and_wrap()
            return

    # ---------------------------
    # Action plan & wrap-up
    # ---------------------------
    def create_action_plan_and_wrap(self):
        print("\n🗺️ Short action plan (4-5 sharp steps):")
        print("- Ground, hydrate, rest, short journaling.")
        print("Summary: review recent conversation and try first step now.")
        self.user_profile["problem_collected"] = False
        self.user_profile["problem_collected_texts"] = []
        self.user_profile["problem_phase_counter"] = 0
        self.user_profile["message_count"] = 0
        self.user_profile["conversation_stage"] = "companion"

    # ---------------------------
    # Escalation helpers
    # ---------------------------
    def escalate_to_human(self):
        print("\n--- Human Escalation ---")
        print("If in immediate danger, call your local emergency number now.")

    # ---------------------------
    # Run once programmatic call
    # ---------------------------
    def run_once_text(self, text):
        self.handle_user_input(text)
