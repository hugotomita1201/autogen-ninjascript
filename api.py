# api.py (Final, Optimized, Working Version)

import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


# --- FastAPI App Setup ---
app = FastAPI(title="Direct API Conversational API")

# --- CORS Middleware ---
origins = [
    "http://localhost:8080",
    "https://ninjascript-frontend.onrender.com",  # <-- CONFIRM THIS IS YOUR FRONTEND URL
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
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(
        f"CRITICAL ERROR: Could not configure Google GenAI. Check GEMINI_API_KEY. Details: {e}"
    )

# --- System Prompts & Models (Initialized once on startup) ---
ORCHESTRATOR_SYSTEM_PROMPT = """You are a methodical trading strategy assistant. Your entire purpose is to guide a user through a checklist, one item at a time, to build a complete trading strategy. Your Core Rules: - You MUST ask only one single question in each response. - Your responses must be short and focused on that single question. - NEVER assume the user's answer. - NEVER move on to the next topic until you have received a clear answer for the current one. - Do not greet the user after the first message. Get straight to the next question. Your Checklist Workflow: 1. Your first goal is to understand the Entry and Exit Logic. Ask a single, open-ended question like, "What is the logic for entering and exiting a trade?" and then STOP. 2. Once the user provides the logic, your next goal is to ask about Position Sizing. Ask a single question like, "Got it. How should position size be determined for each trade?" and then STOP. 3. Once the user provides the sizing, your next goal is to ask about the Stop-Loss. Ask a single question like, "Understood. What is the rule for the stop-loss?" and then STOP. 4. Once the user provides the stop-loss, your next goal is to ask about the Profit Target. Ask a single question like, "Okay. And should there be a profit target?" and then STOP. 5. Once you have all four pieces of information, your final goal is to summarize the strategy. --- SUMMARY FORMATTING RULES --- - You must present the summary as a numbered list. - Each numbered item must be on a new line. - The title for each item must be in ALL CAPS, followed by a colon (e.g., "ENTRY:", "SIZING:"). - Do not use any Markdown formatting like asterisks (*). 6. After the perfectly formatted summary, ask for confirmation with a simple question like, "Does this look correct? If so, I will generate the code." 7. If the user confirms, you MUST end your response with the exact phrase: 'GENERATE_THE_CODE'"""

CODER_SYSTEM_PROMPT = """You are a master NinjaScript 8 programmer who strictly follows templates. You will be given a C# strategy template and a summary of trading rules. Your ONLY job is to insert the required code into the template's placeholder sections. The placeholders are: - `AI_GENERATED_STRATEGY_NAME`: Replace this with a suitable class name based on the strategy rules (e.g., EmaCrossoverRsiStrategy). - `// --- AI will add User Inputs / Properties here ---`: Add the necessary `[NinjaScriptProperty]` inputs (e.g., for EMA periods, RSI, etc.). - `// --- AI will initialize indicators here ---`: Add the C# code to initialize the indicators (e.g., `EMA1 = EMA(FastEmaPeriod);`). - `// --- AI: ENTRY LOGIC ---`: Insert the C# code that checks for the entry conditions and executes the `EnterLong()` or `EnterShort()` command. - `// --- AI: EXIT LOGIC ---`: Insert the C# code that checks for all exit conditions (stop-loss, profit-target, reversal signals) and executes `ExitLong()`, `ExitShort()`, or `SetStopLoss()`. You can also add code to other sections like `OnExecutionUpdate` or `State.Configure` if required by the strategy. CRITICAL RULES: - DO NOT change any part of the template that is not a placeholder section. - DO NOT remove the helper methods like `Log()` that are already in the template. - Your final output must be the complete, modified C# code file and nothing else. - Wrap the final code in ```csharp ... ```."""

# Initialize models once to be reused.
# Using the fast 'flash' model for both conversation and coding.
orchestrator_model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
coder_model = genai.GenerativeModel(
    "gemini-2.5-flash-preview-04-17", system_instruction=CODER_SYSTEM_PROMPT
)


# --- API Endpoints ---
@app.post("/start-chat")
def start_chat(request: ChatRequest):
    session_id = str(uuid.uuid4())

    history = [
        {"role": "user", "parts": [ORCHESTRATOR_SYSTEM_PROMPT]},
        {
            "role": "model",
            "parts": [
                "Understood. I will guide you to build your strategy. Let's begin. What are the entry and exit rules?"
            ],
        },
    ]

    chat_session = orchestrator_model.start_chat(history=history)
    response = chat_session.send_message(request.message)
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

    response = chat_session.send_message(request.message)

    if "GENERATE_THE_CODE" in response.text.upper():
        summary = response.text.replace("GENERATE_THE_CODE", "").strip()

        try:
            with open("strategy_template.cs", "r") as f:
                template_code = f.read()
        except FileNotFoundError:
            return {
                "reply": "Error: The strategy_template.cs file was not found on the server.",
                "session_id": None,
            }

        coder_prompt = f"Here is the strategy summary:\n{summary}\n\nHere is the C# template you must use:\n```csharp\n{template_code}\n```\nPlease fill in the template according to your instructions."

        # Call the pre-initialized, faster coder_model
        final_code_response = coder_model.generate_content(coder_prompt)

        del chat_sessions[request.session_id]
        return {"reply": final_code_response.text, "session_id": None}
    else:
        return {"session_id": request.session_id, "reply": response.text}
