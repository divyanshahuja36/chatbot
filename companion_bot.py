
"""
Companion Bot - Final version focused problem collector & sharp wrap-up
install these pip install openai SpeechRecognition pyaudio TextBlob
Features:
 - ENTER to toggle background speech listening (if microphone available)
 - Typed input fallback
 - Collects user's primary problem, then focuses replies specifically on that problem
 - Relationship (breakup/cheating) and job-betrayal flows have tailored advice
 - Detects suicidal language and asks "how long" — prompts assessment/escalation if needed
 - Suggests PHQ-9 / GAD-7 when duration or sentiment suggests
 - Problem-phase limited to a few focused steps (configurable)
 - Wrap-up summary & 4-5 actionable plan at wind-up threshold (30-40 messages)
 - Minimal hardcoded keys per user instruction (replace with env vars if desired)
"""

import re
import threading
from datetime import datetime
from queue import Queue, Empty

import speech_recognition as sr
from textblob import TextBlob

# Try to import AzureOpenAI; if not available, code will use stubbed responses.
try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except Exception:
    AZURE_AVAILABLE = False

class CompanionBot:
    def __init__(self,
                 checkin_interval_seconds=60 * 60 * 4,
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

        # Speech recognizer (best-effort)
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
        except Exception as e:
            print("[Warning] Microphone init failed (type instead):", e)
            self.recognizer = None
            self.microphone = None

        # Background audio queue & control
        self._bg_text_queue = Queue()
        self._stop_bg_listening = None
        self._listening_lock = threading.Lock()
        self.is_listening = False

        # Session state
        self.user_profile = {
            "age": None,
            "occupation": None,
            "main_concerns": [],
            "problem_type": None,  # 'relationship', 'job', 'other'
            "problem_collected": False,
            "problem_collected_texts": [],
            "conversation_stage": "companion",  # companion, assessment, wrap_up
            "message_count": 0,
            "sentiment_history": [],
            "risk_level": "low",
            "last_assistant_reply": None,
            "duration_flagged": False,
            "problem_phase_counter": 0
        }

        self.conversation_history = []
        self.problem_phase_limit = problem_phase_limit  # e.g., 4 focused steps
        self.wrap_up_threshold = wrap_up_threshold    # e.g., 35 messages
        self.checkin_interval_seconds = checkin_interval_seconds

        # Detection maps
        self.risk_words = {
            "severe": ["suicide", "kill myself", "end it all", "no point living", "better off dead"],
            "high": ["hopeless", "worthless", "hate myself", "can't go on", "everything is wrong"],
            "moderate": ["depressed", "anxious", "panic", "scared", "overwhelmed", "stressed"],
            "low": ["tired", "worried", "sad", "down", "upset"]
        }

        # Check-in scheduler
        self._stop_checkins = threading.Event()
        self._checkin_thread = None

        # Shutdown flag
        self._shutdown_event = threading.Event()

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
        # Simple parser for durations like "for 2 weeks", "since last month", "for three months"
        text = text.lower()
        # numeric patterns
        m = re.search(r"for\s+(\d+)\s*(day|days|week|weeks|month|months|year|years)", text)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            if "day" in unit:
                return n
            if "week" in unit:
                return n * 7
            if "month" in unit:
                return n * 30
            if "year" in unit:
                return n * 365
        # word numbers
        word_nums = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        m2 = re.search(r"for\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s*(day|week|month|year)s?", text)
        if m2:
            n = word_nums.get(m2.group(1), 1)
            unit = m2.group(2)
            if "day" in unit:
                return n
            if "week" in unit:
                return n*7
            if "month" in unit:
                return n*30
            if "year" in unit:
                return n*365
        # since last week/month
        if "since last week" in text:
            return 7
        if "since last month" in text:
            return 30
        if "since yesterday" in text:
            return 1
        return None

    def detect_problem_type(self, text):
        # Relationship keywords
        rl = text.lower()
        relationship_keywords = ["breakup", "broke up", "cheat", "cheated", "girlfriend", "boyfriend", "partner", "relationship", "cheating"]
        job_keywords = ["fired", "laid off", "lost my job", "lost job", "betray", "boss", "coworker", "job", "workplace", "resign", "quit", "sacked"]
        if any(k in rl for k in relationship_keywords):
            return "relationship"
        if any(k in rl for k in job_keywords):
            return "job"
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
        # Short, targeted templates depending on role_hint and problem_type
        pt = self.user_profile.get("problem_type") or "other"
        if role_hint == "companion":
            if pt == "relationship":
                return ("I’m so sorry that happened. You didn’t deserve betrayal—your worth is not defined by their choices. "
                        "Focus on small safety steps right now: breathe slowly, sip water, and let yourself feel what comes. "
                        "If you'd like, I can suggest 3 small things to do in the next hour to feel slightly steadier.")
            if pt == "job":
                return ("That’s a painful betrayal at work — being treated unfairly or losing a job can shake your sense of safety. "
                        "Start with small steps: take a break, breathe, and if possible write down what happened in one short paragraph. "
                        "I can suggest next steps for your career and emotional recovery.")
            # generic
            return ("I hear you. That sounds really hard. I can sit with you, suggest grounding exercises, or help with a short plan to get through the next few hours.")
        if role_hint == "wrap_up":
            return ("Here’s a short 4-step plan to move forward: 1) Ground & stabilize now 2) Short self-care (food/water/rest) 3) Journaling or assessment 4) Reach out / schedule follow-up.")
        if role_hint == "assessment_prompt":
            return ("It might help to do a brief screening like PHQ-9 (depression) or GAD-7 (anxiety). Would you like to try one now?")
        return "I’m here with you."

    def call_ai(self, user_input, role_hint="companion", extra_system=""):
        # Use Azure if available, otherwise fallback
        if not self.client:
            return self._ai_stub(user_input, role_hint, extra_system)
        try:
            system_prompt = "You are a warm, empathetic AI companion. Be concise and supportive."
            if role_hint == "wrap_up":
                system_prompt += " Provide a sharp 4-5 step action plan tailored to the user's recent messages."
            messages = [{"role":"system", "content": system_prompt}]
            # add a few recent conversation turns for context
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
    # Assessments: PHQ-9 and GAD-7 (interactive)
    # ---------------------------
    def run_phq9(self):
        print("\n--- PHQ-9: Over the last 2 weeks, how often have you been bothered by the following?")
        questions = [
            "1. Little interest or pleasure in doing things?",
            "2. Feeling down, depressed, or hopeless?",
            "3. Trouble falling or staying asleep, or sleeping too much?",
            "4. Feeling tired or having little energy?",
            "5. Poor appetite or overeating?",
            "6. Feeling bad about yourself — or that you are a failure?",
            "7. Trouble concentrating on things, such as reading or watching TV?",
            "8. Moving or speaking so slowly that other people notice, or the opposite — being fidgety?",
            "9. Thoughts that you would be better off dead or hurting yourself?"
        ]
        score = 0
        for q in questions:
            while True:
                ans = input(f"{q}\n(0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day): ").strip()
                if ans in {"0","1","2","3"}:
                    score += int(ans); break
                print("Enter 0,1,2 or 3.")
        print(f"PHQ-9 score: {score}/27")
        self.user_profile["last_assessment"] = {"type":"PHQ-9", "score":score, "time": datetime.now().isoformat()}
        if score >= 20:
            print("⚠️ Severe depressive symptoms. Recommend immediate professional support.")
            self.escalate_to_human()
        elif score >= 15:
            print("🔔 Moderately severe — consider contacting a clinician.")
        elif score >= 10:
            print("🔎 Moderate — consider monitoring and maybe professional input.")
        else:
            print("ℹ️ Mild/minimal — practice self-care and monitor.")

    def run_gad7(self):
        print("\n--- GAD-7: Over the last 2 weeks, how often have you been bothered by the following?")
        questions = [
            "1. Feeling nervous, anxious, or on edge?",
            "2. Not being able to stop or control worrying?",
            "3. Worrying too much about different things?",
            "4. Trouble relaxing?",
            "5. Being so restless it's hard to sit still?",
            "6. Becoming easily annoyed or irritable?",
            "7. Feeling afraid as if something awful might happen?"
        ]
        score = 0
        for q in questions:
            while True:
                ans = input(f"{q}\n(0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day): ").strip()
                if ans in {"0","1","2","3"}:
                    score += int(ans); break
                print("Enter 0,1,2 or 3.")
        print(f"GAD-7 score: {score}/21")
        self.user_profile["last_assessment"] = {"type":"GAD-7", "score":score, "time": datetime.now().isoformat()}
        if score >= 15:
            print("⚠️ Severe anxiety — consider immediate professional support.")
            self.escalate_to_human()
        elif score >= 10:
            print("🔔 Moderate — consider seeking support.")
        else:
            print("ℹ️ Mild — self-care recommended.")

    def prompt_assessment(self):
        print("\nWould you like to try a brief screening: 1) PHQ-9 (depression), 2) GAD-7 (anxiety), or 'skip'?")
        ch = input("Choice (1/2/skip): ").strip().lower()
        if ch == "1": self.run_phq9()
        elif ch == "2": self.run_gad7()
        else: print("Okay — we can do it later.")

    # ---------------------------
    # Escalation helpers
    # ---------------------------
    def escalate_to_ai_psychologist(self):
        print("\n[Connecting to AI psychologist — simulated]")
        reply = "AI Psychologist: Based on the info, consider immediate grounding and a short structured check-in tomorrow. Would you like to book? (simulated)"
        self.conversation_history.append({"user":"[escalate_ai]", "assistant":reply, "sentiment":None, "stage":"escalation", "timestamp":datetime.now().isoformat()})
        print(reply)

    def escalate_to_human(self):
        print("\n--- Human Escalation ---")
        print("If you are in immediate danger, call your local emergency number now.")
        print("If in the US, call 988 for suicide & crisis lifeline. Contact a trusted local clinician or nearest ER.")
        # Placeholder for action to connect to human ops

    # ---------------------------
    # Problem-phase: collect + focused replies (bounded)
    # ---------------------------
    def start_problem_collection_if_needed(self, text):
        # If problem not yet collected, detect and start collecting
        if not self.user_profile["problem_collected"]:
            ptype = self.detect_problem_type(text)
            self.user_profile["problem_type"] = ptype
            self.user_profile["problem_collected"] = True
            self.user_profile["problem_collected_texts"].append(text)
            # First focused reply
            return True
        return False

    def focused_companion_reply(self, user_input, sentiment):
        # increment problem-phase counter
        self.user_profile["problem_phase_counter"] += 1
        ptype = self.user_profile.get("problem_type", "other")
        # If suicidal language occurs, handle immediately
        if self.detect_suicidal_language(user_input):
            # ask how long they've felt this way (critical)
            print("\n⚠️ You mentioned suicidal thoughts. I'm really sorry you're feeling so overwhelmed.")
            duration = input("How long have you felt like this? (e.g., 'a few days', '2 weeks', 'months'): ").strip()
            days = self.parse_duration_days(duration) or None
            if days and days >= 14:
                print("Because this has been going on for a while, I strongly recommend a PHQ-9 / GAD-7 and immediate support.")
                self.prompt_assessment()
            print("If you are in immediate danger, please call emergency services or a crisis line now (988 in US).")
            self.escalate_to_human()
            return

        # Choose tailored reply (use LLM if available)
        role_hint = "companion"
        if self.client:
            ai_reply = self.call_ai(user_input, role_hint)
        else:
            # Templates tuned per problem type
            if ptype == "relationship":
                ai_reply = ("I’m so sorry. That was a betrayal, and you didn’t deserve it. Right now, focus on one small thing: "
                            "breathe slowly for the next 2 minutes and put on something that comforts you. Afterwards, if you want, "
                            "we can outline 3 things to do in the next 24 hours to support yourself.")
            elif ptype == "job":
                ai_reply = ("That’s a really unfair situation — workplaces can be brutal. Take a short break, write down what happened in one paragraph, "
                            "and then we can map immediate steps: emotional stabilization, documenting the event, and exploring next job options.")
            else:
                ai_reply = self._ai_stub(user_input, "companion")

        # Show & store
        self.conversation_history.append({"user": user_input, "assistant": ai_reply, "sentiment": sentiment, "stage":"companion", "timestamp": datetime.now().isoformat()})
        self.user_profile["message_count"] += 1
        self.display(ai_reply, "companion", sentiment)

        # If problem-phase exceeded limit -> create plan & wrap
        if self.user_profile["problem_phase_counter"] >= self.problem_phase_limit:
            self.create_action_plan_and_wrap()
        return

    # ---------------------------
    # Action plan (sharp 4-5 steps) and wrap-up
    # ---------------------------
    def create_action_plan_and_wrap(self):
        # Build a tailored 4-5 step plan based on problem_type and risk
        ptype = self.user_profile.get("problem_type", "other")
        rl = self.user_profile.get("risk_level", "low")
        recent = [e["user"] for e in self.conversation_history[-6:]]
        recent_summary = " ".join(recent[-3:]) if recent else "No recent details."

        steps = []
        # 1: stabilize
        steps.append("1) Immediate grounding: 5 slow breaths (4in-4out), sip water, sit or lie down somewhere safe.")
        # 2: safety & body care
        steps.append("2) Body care in next hour: hydrate, eat a small snack, put on comfy clothes, rest if needed.")
        # 3: targeted step
        if ptype == "relationship":
            steps.append("3) Emotional processing: write a short letter (you don't need to send it) describing what happened and how it felt.")
            steps.append("4) Social support: message one trusted person 'I need a bit of support right now' and set a time to talk.")
        elif ptype == "job":
            steps.append("3) Practical step: write down the facts (dates, people, what happened) — keep it for your records.")
            steps.append("4) Career step: if possible, update one line on your resume or LinkedIn, or look at 1 job posting to begin momentum.")
        else:
            steps.append("3) Processing: 10-minute free journaling about what you feel and one small positive moment today.")
            steps.append("4) Follow-up: schedule a 10-minute check-in with yourself or a friend tomorrow.")

        # If risk high/severe add escalation
        if rl in ("high", "severe"):
            steps.append("5) Mental health: consider taking PHQ-9/GAD-7 now and contacting a clinician. For immediate danger call emergency services / crisis line.")

        # Print action plan
        print("\n" + "="*60)
        print("🗺️ Short action plan (4-5 sharp steps):")
        for s in steps:
            print("-", s)
        print("\nRecent summary:", recent_summary)
        print("="*60)

        # Wrap up - store final assistant message and reset relevant counters
        wrap_msg = "I summarized 4-5 concrete steps above. Try the first one now (grounding). If things are severe, please get immediate help."
        self.conversation_history.append({"user":"[action_plan]", "assistant":wrap_msg, "sentiment":None, "stage":"wrap_up", "timestamp": datetime.now().isoformat()})
        self.display(wrap_msg, "wrap_up", {"polarity":0.0, "risk_level": self.user_profile.get("risk_level","low")})

        # Reset session for next check-in while preserving history
        self.user_profile["problem_collected"] = False
        self.user_profile["problem_collected_texts"] = []
        self.user_profile["problem_phase_counter"] = 0
        self.user_profile["message_count"] = 0
        self.user_profile["conversation_stage"] = "companion"

    # ---------------------------
    # Display helper (avoid repeats)
    # ---------------------------
    def display(self, text, stage, sentiment):
        # avoid exact duplicate assistant replies
        last = self.user_profile.get("last_assistant_reply")
        if last and last.strip() == text.strip():
            # give a minimal variant
            alt = "I hear you. I'm here. If you want, we can try a grounding exercise or make a simple plan."
            text = alt
        print("\n" + "="*60)
        print(f"🤖 Companion ({stage.title()}):")
        print(text)
        mood_emo = "😊" if sentiment["polarity"] > 0.1 else "😐" if sentiment["polarity"] > -0.1 else "😔"
        risk_map = {"low":"💚","moderate":"💛","high":"🧡","severe":"🔴"}
        print(f"\n📊 Mood: {mood_emo} | Risk: {risk_map.get(sentiment['risk_level'],'💚')}")
        print("="*60)
        self.user_profile["last_assistant_reply"] = text

    # ---------------------------
    # Background speech callback & ENTER toggle
    # ---------------------------
    def _bg_callback(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(audio)
            self._bg_text_queue.put((datetime.now().isoformat(), text))
        except sr.UnknownValueError:
            pass
        except Exception as e:
            print("[speech callback error]", e)

    def toggle_listen_enter_flow(self):
        with self._listening_lock:
            if self.is_listening:
                # stop
                if self._stop_bg_listening:
                    try:
                        self._stop_bg_listening(wait_for_stop=False)
                    except TypeError:
                        self._stop_bg_listening()
                self.is_listening = False
                print("🔴 Stopped listening. Processing collected audio...")
                texts = []
                while True:
                    try:
                        ts, t = self._bg_text_queue.get_nowait()
                        texts.append((ts, t))
                    except Empty:
                        break
                full = " ".join(t for _, t in texts).strip()
                if full:
                    print(f"\n👤 You said (compiled): {full}\n")
                    self.handle_user_input(full)
                else:
                    print("No clear speech captured. Try again.")
                return

            # start
            if not self.recognizer or not self.microphone:
                print("Microphone not available. Use typed input.")
                return
            try:
                print("⚪ Press ENTER when you want to STOP listening.")
                while not self._bg_text_queue.empty():
                    try: self._bg_text_queue.get_nowait()
                    except Exception: break
                stop = self.recognizer.listen_in_background(self.microphone, self._bg_callback)
                self._stop_bg_listening = stop
                self.is_listening = True
                print("🟢 Listening in background — speak now. (Press ENTER to stop)")
            except Exception as e:
                print("[listen error]", e)

    # ---------------------------
    # Main pipeline for user input
    # ---------------------------
    def handle_user_input(self, user_input):
        # analyze sentiment
        sentiment = self.analyze_sentiment(user_input)

        # record message for history; assistant reply added later
        # detect suicidal language first
        if self.detect_suicidal_language(user_input):
            # immediate crisis flow
            print("\n⚠️ You used language that suggests severe distress. Please tell me: how long have you felt this way?")
            dur = input("Duration (e.g., 'a few days', '2 weeks'): ").strip()
            days = self.parse_duration_days(dur) or None
            if days and days >= 14:
                print("Because this has lasted a while, I recommend taking a short assessment (PHQ-9 / GAD-7) and reaching out to a clinician.")
                self.prompt_assessment()
            print("If you feel you might act on these thoughts, contact emergency services or a crisis line immediately (988 in US).")
            self.escalate_to_human()
            # store and return
            self.conversation_history.append({"user":user_input,"assistant":"[crisis_handled]","sentiment":sentiment,"stage":"crisis","timestamp":datetime.now().isoformat()})
            return

        # If problem not collected yet, collect & start problem-phase
        if not self.user_profile["problem_collected"]:
            # detect and mark
            ptype = self.detect_problem_type(user_input)
            self.user_profile["problem_type"] = ptype
            self.user_profile["problem_collected"] = True
            self.user_profile["problem_collected_texts"].append(user_input)
            # detect duration mention and flag assessment suggestion
            d = self.parse_duration_days(user_input)
            if d and d >= 14:
                self.user_profile["duration_flagged"] = True
            # initial focused reply
            self.user_profile["problem_phase_counter"] = 0
            # produce focused reply
            self.focused_companion_reply(user_input, sentiment)
            return

        # If already collecting problem and in problem-phase, give focused replies until limit
        if self.user_profile["problem_collected"]:
            if self.user_profile["problem_phase_counter"] < self.problem_phase_limit:
                # continue focused replies
                self.focused_companion_reply(user_input, sentiment)
                return
            else:
                # problem-phase exhausted -> create action plan & wrap
                self.create_action_plan_and_wrap()
                return

        # default companion flow
        ai_reply = self.call_ai(user_input, role_hint="companion")
        self.conversation_history.append({"user":user_input,"assistant":ai_reply,"sentiment":sentiment,"stage":"companion","timestamp":datetime.now().isoformat()})
        self.user_profile["message_count"] += 1
        self.display(ai_reply, "companion", sentiment)

        # Suggest assessment if many negative sentiments or duration flagged
        recent = self.user_profile["sentiment_history"][-5:]
        neg_count = sum(1 for s in recent if s["polarity"] < -0.15)
        if neg_count >= 3 or self.user_profile.get("duration_flagged"):
            print("\nℹ️ It might be helpful to try a short assessment (PHQ-9 or GAD-7). Would you like to now?")
            confirm = input("(yes/no): ").strip().lower()
            if confirm in ("yes","y"):
                self.prompt_assessment()

        # wind-up if message_count reaches threshold
        if self.user_profile["message_count"] >= self.wrap_up_threshold:
            print("\n✨ We've reached the session wrap-up threshold. I'll produce a short summary and next steps.")
            self.summarize_and_wrap()

    # ---------------------------
    # Summary & wrap
    # ---------------------------
    def summarize_and_wrap(self):
        n = 30
        last = self.conversation_history[-n:]
        print("\n" + "="*60)
        print("📝 Session summary (recent):")
        for e in last:
            print("You:", e["user"])
            print("Bot:", e["assistant"])
            s = e.get("sentiment")
            if s:
                print(f"  Mood: {s['polarity']:.2f} | Risk: {s['risk_level']}")
            print("-"*40)
        rl = self.user_profile.get("risk_level", "low")
        print("\nShort next steps:")
        if rl == "low":
            print("- Grounding, hydrate, rest, short journaling.")
        elif rl == "moderate":
            print("- Guided breathing, journaling, consider PHQ-9/GAD-7, reach out to a friend.")
        elif rl == "high":
            print("- Take PHQ-9/GAD-7, schedule a clinician, consider escalation to AI/human.")
        else:
            print("- URGENT: contact emergency services or crisis line now.")
        print("="*60)
        # reset counters for next session but keep history
        self.user_profile["message_count"] = 0
        self.user_profile["problem_collected"] = False
        self.user_profile["problem_collected_texts"] = []
        self.user_profile["problem_phase_counter"] = 0
        self.user_profile["conversation_stage"] = "companion"

    # ---------------------------
    # Check-in scheduler
    # ---------------------------
    def _checkin_loop(self):
        while not self._stop_checkins.is_set() and not self._shutdown_event.is_set():
            stopped = self._stop_checkins.wait(timeout=self.checkin_interval_seconds)
            if stopped or self._shutdown_event.is_set():
                break
            print("\n💬 Check-in: How are you feeling right now? (Press ENTER to speak)")
            input()
            self.toggle_listen_enter_flow()

    def start_checkins(self):
        if self._checkin_thread and self._checkin_thread.is_alive(): return
        self._stop_checkins.clear()
        self._checkin_thread = threading.Thread(target=self._checkin_loop, daemon=True)
        self._checkin_thread.start()
        print(f"[Scheduler] Check-ins every {self.checkin_interval_seconds} seconds started.")

    def stop_checkins(self):
        self._stop_checkins.set()
        if self._checkin_thread:
            self._checkin_thread.join(timeout=1)

    # ---------------------------
    # Main loop
    # ---------------------------
    def run(self):
        print("🌟 Welcome to Your AI Psychology Companion! 🌟")
        print("Mode: ENTER to toggle listening. Type 'quit' anytime to exit.")
        print("Commands: /checkin_start, /checkin_stop, /escalate_ai, /escalate_human\n")

        try:
            while not self._shutdown_event.is_set():
                cmd = input("Press ENTER to start/stop speaking, or type a command/message: ").strip()
                if cmd.lower() in ("quit","exit"):
                    print("🤖 Thank you for sharing today. Take care.")
                    break
                if cmd == "":
                    self.toggle_listen_enter_flow()
                    continue
                if cmd == "/checkin_start":
                    self.start_checkins(); continue
                if cmd == "/checkin_stop":
                    self.stop_checkins(); continue
                if cmd == "/escalate_ai":
                    self.escalate_to_ai_psychologist(); continue
                if cmd == "/escalate_human":
                    self.escalate_to_human(); continue

                # typed message
                self.handle_user_input(cmd)

        except KeyboardInterrupt:
            print("\n[Interrupted]")
        finally:
            self.shutdown()

    def shutdown(self):
        self._shutdown_event.set()
        try:
            if self._stop_bg_listening:
                self._stop_bg_listening(wait_for_stop=False)
        except Exception:
            pass
        self.stop_checkins()
        print("💤 Companion shut down. Goodbye.")

    # programmatic single call
    def run_once_text(self, text):
        self.handle_user_input(text)

