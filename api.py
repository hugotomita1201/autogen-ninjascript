# api.py (Final, Robust Version)

import os
import uuid
import autogen
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


# --- FastAPI App Setup ---
app = FastAPI(title="Autogen Conversational API")

# --- CORS Middleware ---
# This block allows your frontend to communicate with your backend.
origins = [
    "http://localhost:8080",
    "https://ninjascript-frontend.onrender.com",  # <-- MAKE SURE THIS IS YOUR CORRECT RENDER FRONTEND URL
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

# --- Autogen Configuration ---
# We are using the "google" api_type as it's the valid one for Autogen's config,
# and we are relying on the GOOGLE_CLOUD_PROJECT environment variable to prevent the crash.
config_list = [
    {
        "model": "gemini-1.5-pro-latest",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "api_type": "google",
    }
]
llm_config = {"config_list": config_list, "timeout": 180}


# --- Agent Definitions ---
def is_termination_msg(content):
    have_content = content.get("content", "") is not None
    if have_content:
        if "GENERATE_THE_CODE" in content["content"].upper():
            return True
    return False


# This agent guides the user through the checklist
orchestrator = autogen.AssistantAgent(
    name="Orchestrator",
    llm_config=llm_config,
    system_message="""You are a lead trading strategist. Your goal is to guide the user to build a complete trading strategy by asking one question at a time.
    1. First, get the entry/exit logic.
    2. Then, get the position sizing.
    3. Then, get the stop-loss.
    4. Then, get the profit target.
    5. Finally, present a complete, numbered summary of all collected rules.
    After presenting the summary, ask the user 'Does this look correct? If so, I will generate the code.'
    If the user confirms, you MUST end your response with the exact phrase: 'GENERATE_THE_CODE'""",
)

# This agent's only job is to write the final code
coder_agent = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message="""You are an expert NinjaScript 8 programmer. You will be given a complete summary of a trading strategy.
    Your only job is to write the full, complete, and valid NinjaScript C# code based on that summary.
    Wrap the final code in ```csharp ... ```.""",
)

# --- API Endpoints (New Simplified Logic) ---


@app.post("/start-chat")
def start_chat(request: ChatRequest):
    session_id = str(uuid.uuid4())

    # Store history for this new session
    chat_sessions[session_id] = []

    # Manually start the conversation by sending the user's first message to the Orchestrator
    user_message = {"role": "user", "content": request.message}
    chat_sessions[session_id].append(user_message)

    response = orchestrator.generate_reply(messages=chat_sessions[session_id])

    # Save the orchestrator's first response to the history
    assistant_message = {"role": "assistant", "content": response}
    chat_sessions[session_id].append(assistant_message)

    return {"session_id": session_id, "reply": response}


@app.post("/continue-chat")
def continue_chat(request: ChatRequest):
    session_history = chat_sessions.get(request.session_id)
    if not session_history:
        return {
            "reply": "Error: Chat session not found. Please refresh and start a new chat.",
            "session_id": None,
        }

    # Append the user's new message to the history
    user_message = {"role": "user", "content": request.message}
    session_history.append(user_message)

    # Get the next reply from the orchestrator based on the *entire* conversation history
    response = orchestrator.generate_reply(messages=session_history)

    # Check for the termination trigger phrase
    if is_termination_msg({"content": response}):
        summary = response.replace("GENERATE_THE_CODE", "").strip()

        # Call the Coder agent with the final summary
        final_code = coder_agent.generate_reply(
            messages=[
                {
                    "role": "user",
                    "content": f"Please generate the NinjaScript code for the following strategy: {summary}",
                }
            ]
        )

        # Clean up the completed session and return the final code
        del chat_sessions[request.session_id]
        return {"reply": final_code, "session_id": None}
    else:
        # Save the new assistant message to history and continue the chat
        assistant_message = {"role": "assistant", "content": response}
        session_history.append(assistant_message)
        return {"session_id": request.session_id, "reply": response}
