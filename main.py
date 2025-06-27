import autogen

# Step 1: Configure your LLM (No changes here)
config_list = [
    {
        "model": "gemini-2.5-flash-preview-04-17",
        "api_key": "AIzaSyCfAnsdMO1D02ghuaPc-ny1Vu9q6hyOGZA",
        "api_type": "google",
    }
]
llm_config = {"config_list": config_list, "timeout": 120}


# Step 2: Define Your Agents (with the stricter Rule_Setter)

# User Proxy Agent
user_proxy_agent = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="A human admin.",
    code_execution_config=False,
    human_input_mode="NEVER",
)

# Rule Setter Agent (with the improved prompt)
rule_setter_agent = autogen.AssistantAgent(
    name="Rule_Setter",
    llm_config=llm_config,
    system_message="""You are a literal trading rule interpreter.
    Your only job is to translate the user's exact request into a formal list.
    **You must not add any new rules, indicators, or concepts that the user did not explicitly mention.**
    If a component like a stop-loss, profit-target, or position size is missing, you must state that it is undefined.
    Your output is only the formalized list of rules. After the list, you must say 'TERMINATE'.""",
)

# Sizer Agent
sizer_agent = autogen.AssistantAgent(
    name="Sizer",
    llm_config=llm_config,
    system_message="""You are an expert in risk management. Your only job is to formalize the user's request for position sizing and stop-loss.
    You must extract any user requests for lot size, risk amount, or stop-loss type (e.g., ticks, ATR).
    If no sizing is mentioned, you should state that it is undefined.
    State the parameters clearly and then say 'TERMINATE'.""",
)

# Coder Agent
coder_agent = autogen.AssistantAgent(
    name="NinjaScript_Coder",
    llm_config=llm_config,
    system_message="""You are an expert NinjaScript programmer. You only write complete, valid NinjaScript 8 C# code.
    You will receive a set of precise rules from the other agents.
    Your task is to combine all the rules and generate a single, complete NinjaScript strategy file.
    Do not add any features not specified. Start your response with '```csharp' and end it with '```'.
    After you have provided the complete code, you must say 'TERMINATE'.""",
)


# Step 3: Define the "Playbook" (Custom Speaker Selection Function)
def speaker_selection_method(last_speaker, groupchat):
    """
    This function acts as our playbook. It enforces a strict order of speakers.
    """
    # The conversation always starts with the User Proxy.
    # The manager will select the next speaker based on the last one.
    if last_speaker is user_proxy_agent:
        # User -> Rule_Setter
        return rule_setter_agent
    elif last_speaker is rule_setter_agent:
        # Rule_Setter -> Sizer
        return sizer_agent
    elif last_speaker is sizer_agent:
        # Sizer -> Coder
        return coder_agent
    elif last_speaker is coder_agent:
        # The coder was the last one to speak, so the task is done.
        # Returning None ends the conversation.
        return None


# Step 4: Set up the Group Chat with the Playbook
groupchat = autogen.GroupChat(
    agents=[user_proxy_agent, rule_setter_agent, sizer_agent, coder_agent],
    messages=[],
    max_round=10,
    speaker_selection_method=speaker_selection_method,  # <-- We are plugging in our playbook here!
)

# Create the manager for the group chat.
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


# Step 5: Initiate the Chat
user_proxy_agent.initiate_chat(
    manager,
    message="""
    I need a futures strategy based on two EMAs and an RSI.
    The entry rule is to go long when the 12-period EMA crosses above the 26-period EMA, but only if the 14-period RSI is above 50.
    The exit rule is when the 12-period EMA crosses below the 26-period EMA.
    For risk, use a fixed size of 2 contracts and a static 25-tick stop-loss.
    """,
)
