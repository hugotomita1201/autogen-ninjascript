# api.py (Conversational Version)

import os
import uuid
import autogen
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# --- Pydantic Models for API Data Structure ---
class ChatRequest(BaseModel):
    session_id: str
    message: str


class StartChatRequest(BaseModel):
    prompt: str


# --- FastAPI App Setup ---
app = FastAPI(title="Autogen Conversational API")

# Add CORS Middleware to allow our frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory "database" to store active chat sessions
# In a production app, you would replace this with a real database like Redis or a SQL DB.
chat_sessions = {}


# The new, specific config
config_list = [
    {
        "model": "gemini-2.5-flash-preview-04-17",
        "api_key": os.getenv("AIzaSyCfAnsdMO1D02ghuaPc-ny1Vu9q6hyOGZA"),
        "api_type": "google",  # <-- Change "google" to "gemini"
    }
]
llm_config = {"config_list": config_list, "timeout": 120}

# --- Agent Definitions ---


# This function will check if the user's message is a termination command.
def is_termination_msg(content):
    have_content = content.get("content", "") is not None
    if have_content:
        if "GENERATE_CODE_NOW" in content["content"]:
            return True  # Terminate if the final trigger is detected
    return False


# USER PROXY AGENT (Configured for interaction)
# human_input_mode is set to "ALWAYS" to ensure it stops for user input.
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="A human user who provides input.",
    human_input_mode="ALWAYS",
    is_termination_msg=is_termination_msg,
    code_execution_config=False,
)

# ORCHESTRATOR AGENT (The new "Master" agent)
# This agent follows a strict checklist to guide the user.
orchestrator = autogen.AssistantAgent(
    name="Orchestrator",
    llm_config=llm_config,
    system_message="""You are a lead trading strategist. Your goal is to guide the user to build a complete trading strategy by filling out a checklist.
    You must ask one question at a time. Do not assume any answers.

    **Your Checklist:**
    1.  **Entry/Exit Logic:** First, ask for the basic entry and exit rules (e.g., "what indicators and conditions should trigger a long entry and exit?").
    2.  **Position Sizing:** After getting the rules, ask about position sizing (e.g., "How many contracts or lots should be traded?").
    3.  **Stop-Loss:** After sizing, ask for the stop-loss mechanism (e.g., "Should there be a stop-loss? If so, should it be based on a fixed number of ticks, or something else?").
    4.  **Profit Target:** After the stop-loss, ask for a profit target (e.g., "Should there be a profit target?").
    5.  **Final Confirmation:** Once all items are collected, present a complete, numbered summary of the entire strategy. Ask the user for final confirmation.
    
    If they confirm, you MUST end your response with the single phrase: "GENERATE_CODE_NOW".""",
)

# CODER AGENT (Now triggered at the end)
coder_agent = autogen.AssistantAgent(
    name="NinjaScript_Coder",
    llm_config=llm_config,
    system_message="""You are an expert NinjaScript programmer. You will be given a complete strategy summary.
    Your only job is to write the full, complete, and valid NinjaScript 8 C# code based on that summary.
    Do not add any features not specified. Wrap the final code in ```csharp ... ```.""",
)


# --- API Endpoints ---


@app.post("/start-chat")
def start_chat(request: StartChatRequest):
    session_id = str(uuid.uuid4())

    # We'll use a simple two-agent chat for the interactive part
    groupchat = autogen.GroupChat(
        agents=[user_proxy, orchestrator], messages=[], max_round=20
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Store the manager in our session dictionary
    chat_sessions[session_id] = manager

    # Initiate the chat. The user's first prompt is the first message.
    user_proxy.initiate_chat(manager, message=request.prompt)

    # The chat is now paused, waiting for the next human input.
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

    # Send the user's message to the waiting agent group
    user_proxy.send(message=request.message, recipient=manager)

    # The conversation will run until it needs the next human input or terminates.
    last_message = manager.chat_messages[user_proxy][-1]["content"]

    # Check if the trigger phrase is in the last message
    if "GENERATE_CODE_NOW" in last_message:
        # The interactive part is done. Time to generate the final code.
        # We take the summary from the orchestrator to give to the coder.
        strategy_summary = last_message.replace("GENERATE_CODE_NOW", "").strip()

        # We start a NEW, non-interactive chat with the CoderAgent
        coder_chat_manager = autogen.GroupChatManager(
            groupchat=autogen.GroupChat(agents=[user_proxy, coder_agent], messages=[])
        )
        user_proxy.initiate_chat(coder_chat_manager, message=strategy_summary)

        # Extract the final code from the coder's response
        final_code = coder_chat_manager.chat_messages[user_proxy][-1]["content"]

        # Clean up the completed session
        del chat_sessions[request.session_id]
        return {
            "reply": final_code,
            "session_id": None,
        }  # None ID signals the frontend to end the chat

    # If not finished, just return the next question from the orchestrator
    return {"session_id": request.session_id, "reply": last_message}
