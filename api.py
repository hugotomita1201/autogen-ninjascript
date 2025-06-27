# api.py (The Final, Working Version)

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

# --- Autogen Configuration ---
config_list = [
    {
        "model": "gemini-2.5-flash-preview-04-17",
        "api_key": os.getenv("AIzaSyCfAnsdMO1D02ghuaPc-ny1Vu9q6hyOGZA"),
        "api_type": "google",
    }
]
llm_config = {"config_list": config_list, "timeout": 180}


# --- Agent Definitions ---
def is_termination_msg(content):
    have_content = content.get("content", "") is not None
    if have_content:
        # The User Proxy will terminate the conversation when it sees the final trigger phrase
        if "GENERATE_THE_CODE" in content["content"].upper():
            return True
    return False


# USER PROXY AGENT (Configured for interaction)
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",  # The user's input will be sent programmatically
    is_termination_msg=is_termination_msg,
    code_execution_config=False,
)

# ORCHESTRATOR AGENT
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

# CODER AGENT
coder_agent = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
    system_message="""You are an expert NinjaScript 8 programmer. You will be given a complete summary of a trading strategy.
    Your only job is to write the full, complete, and valid NinjaScript C# code based on that summary.
    Wrap the final code in ```csharp ... ```.""",
)


# --- API Endpoints ---
@app.post("/start-chat")
def start_chat(request: ChatRequest):
    session_id = str(uuid.uuid4())

    groupchat = autogen.GroupChat(
        agents=[user_proxy, orchestrator],
        messages=[],
        max_round=15,
        speaker_selection_method="round_robin",  # Use the simple, predictable round_robin method
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    chat_sessions[session_id] = manager

    # Initiate the chat. The user's first prompt is the first message.
    user_proxy.initiate_chat(manager, message=request.message)

    # The first reply from the orchestrator is the last message in the history.
    reply = groupchat.messages[-1]["content"]

    return {"session_id": session_id, "reply": reply}


@app.post("/continue-chat")
def continue_chat(request: ChatRequest):
    manager = chat_sessions.get(request.session_id)
    if not manager:
        return {
            "reply": "Error: Chat session not found. Please start a new chat.",
            "session_id": None,
        }

    # Send the user's message to the waiting agent group.
    # The user_proxy will continue the conversation from where it left off.
    user_proxy.send(message=request.message, recipient=manager)

    # The conversation will run until it needs the next human input or terminates.
    # The last message in the history will be the agent's next question.
    last_message = manager.chat_messages[user_proxy][-1]

    if is_termination_msg(last_message):
        summary = last_message["content"].replace("GENERATE_THE_CODE", "").strip()
        final_code = coder_agent.generate_reply(
            messages=[
                {
                    "role": "user",
                    "content": f"Generate NinjaScript for this summary: {summary}",
                }
            ]
        )
        del chat_sessions[request.session_id]
        return {"reply": final_code, "session_id": None}
    else:
        return {"session_id": request.session_id, "reply": last_message["content"]}
