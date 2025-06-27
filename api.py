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
genai.configure(api_key=os.getenv("AIzaSyCfAnsdMO1D02ghuaPc-ny1Vu9q6hyOGZA"))

# We define the system prompts for our "conceptual" agents.
ORCHESTRATOR_SYSTEM_PROMPT = """You are a lead trading strategist. Your goal is to guide the user to build a complete trading strategy by asking one question at a time.
1. First, get the entry/exit logic.
2. Then, get the position sizing.
3. Then, get the stop-loss.
4. Then, get the profit target.
5. Finally, present a complete, numbered summary of all collected rules.
After presenting the summary, ask the user 'Does this look correct? If so, I will generate the code.'
If the user confirms, you MUST end your response with the exact phrase: 'GENERATE_THE_CODE'"""

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
