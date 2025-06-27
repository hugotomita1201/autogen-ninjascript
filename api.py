# api.py (Final Version using Direct API Calls)

import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai  # <-- Import the Google library directly


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


# --- FastAPI App Setup ---
app = FastAPI(title="Direct API Conversational API")

origins = [
    "http://localhost:8080",
    "https://ninjascript-frontend.onrender.com",  # <-- MAKE SURE THIS IS YOUR FRONTEND URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory Session Storage ---
chat_sessions = {}

# --- Direct Google Gemini Configuration ---
# We configure the Google client directly, bypassing Autogen's client management.
# CORRECT AND SECURE WAY
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# We define the system prompts for our "conceptual" agents.
# The New, More Methodical Prompt
ORCHESTRATOR_SYSTEM_PROMPT = """You are a methodical trading strategy assistant. Your entire purpose is to guide a user through a checklist, one item at a time, to build a complete trading strategy.

**Your Core Rules:**
- You MUST ask only one single question in each response.
- Your responses must be short and focused on that single question.
- NEVER assume the user's answer or move on to the next topic until you have received a clear answer for the current one.
- Do not greet the user after the first message. Get straight to the next question.

**Your Checklist Workflow:**
1.  Your first goal is to understand the **Entry and Exit Logic**. Ask a single, open-ended question like, "What is the logic for entering and exiting a trade?" and then STOP.
2.  Once the user provides the logic, your next goal is to ask about **Position Sizing**. Ask a single question like, "Got it. How should position size be determined for each trade?" and then STOP.
3.  Once the user provides the sizing, your next goal is to ask about the **Stop-Loss**. Ask a single question like, "Understood. What is the rule for the stop-loss?" and then STOP.
4.  Once the user provides the stop-loss, your next goal is to ask about the **Profit Target**. Ask a single question like, "Okay. And should there be a profit target?" and then STOP.
5.  Once you have all four pieces of information, your final goal is to **summarize the strategy**. Present a complete, numbered summary of every parameter you have collected.
6.  After the summary, ask for confirmation with a simple question like, "Does this look correct? If so, I will generate the code."
7.  If the user confirms, you MUST end your response with the exact phrase: 'GENERATE_THE_CODE'"""

CODER_SYSTEM_PROMPT = """You are an expert NinjaScript 8 programmer. You will be given a complete summary of a trading strategy.
Your only job is to write the full, complete, and valid NinjaScript C# code based on that summary.
Wrap the final code in ```csharp ... ```."""

# Initialize the generative model from Google
# We are using a stable, fast model.
model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

# --- API Endpoints (New Direct API Logic) ---


@app.post("/start-chat")
def start_chat(request: ChatRequest):
    session_id = str(uuid.uuid4())

    # The history starts with the system prompt and the user's first message
    history = [
        {"role": "user", "parts": [ORCHESTRATOR_SYSTEM_PROMPT]},
        {
            "role": "model",
            "parts": [
                "Understood. I will guide you through building your trading strategy. Let's begin. What is the entry and exit logic you have in mind?"
            ],
        },
        {"role": "user", "parts": [request.message]},
    ]

    # Start a chat session directly with the Google API
    chat_session = model.start_chat(history=history)

    # Send the first message to get the first real reply
    response = chat_session.send_message(request.message)

    # Store the chat object itself in our session dictionary
    chat_sessions[session_id] = chat_session

    return {"session_id": session_id, "reply": response.text}


@app.post("/continue-chat")
def continue_chat(request: ChatRequest):
    chat_session = chat_sessions.get(request.session_id)
    if not chat_session:
        return {
            "reply": "Error: Chat session not found. Please refresh and start a new chat.",
            "session_id": None,
        }

    # Send the user's message to the ongoing chat session
    response = chat_session.send_message(request.message)

    # Check for the termination trigger phrase
    if "GENERATE_THE_CODE" in response.text.upper():
        summary = response.text.replace("GENERATE_THE_CODE", "").strip()

        # Start a new, one-shot conversation with the Coder prompt
        coder_history = [
            {"role": "user", "parts": [CODER_SYSTEM_PROMPT]},
            {
                "role": "model",
                "parts": ["Ready to code. Please provide the strategy summary."],
            },
            {
                "role": "user",
                "parts": [
                    f"Please generate the NinjaScript code for this summary: {summary}"
                ],
            },
        ]
        coder_chat = model.start_chat(history=coder_history)
        final_code_response = coder_chat.send_message(
            f"Please generate the NinjaScript code for this summary: {summary}"
        )

        # Clean up the completed session and return the final code
        del chat_sessions[request.session_id]
        return {"reply": final_code_response.text, "session_id": None}
    else:
        # Continue the chat
        return {"session_id": request.session_id, "reply": response.text}
