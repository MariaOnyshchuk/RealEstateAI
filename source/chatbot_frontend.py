import streamlit as st 
import chatbot_backend as backend
from agent import Agent
from router import Router
from summarizer import Summarizer
from agents.base_agent import ParameterExtractor
from agents.base_agent import FamilyAgent, InvestorAgent, YoungProfessionalAgent
import json
import os


def save_feedback_for_kto(user_input, ai_response, is_like):
    data_file = "kto_dataset.json"

    entry = {
        "prompt": user_input,
        "completion": ai_response,
        "label": is_like  
    }
    
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
        
    data.append(entry)
    
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)   

def get_last_user_prompt(history, current_idx):
    for i in range(current_idx - 1, -1, -1):
        if history[i]["role"] == "user":
            return history[i]["text"]
    return None 

def handle_feedback(idx, user_prompt, ai_text, vote_type):
    st.session_state.feedback[idx] = vote_type
    
    if user_prompt:
        is_like = (vote_type == "like")
        save_feedback_for_kto(user_prompt, ai_text, is_like)
    else:
        print(f"Could not find user prompt for message {idx}")

st.title("Real Estate Chatbot 🏠")

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}

# Initialize chat memory
if 'chat_memory' not in st.session_state: 
    st.session_state.chat_memory = backend.create_chat_memory()

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

router = Router(backend._bedrock_llm)
summarizer = Summarizer(backend._bedrock_llm)
extractor = ParameterExtractor(backend._bedrock_llm)

family_agent = FamilyAgent(backend._bedrock_llm)
investor_agent = InvestorAgent(backend._bedrock_llm)
young_professional_agent = YoungProfessionalAgent(backend._bedrock_llm)

agents = [family_agent, investor_agent, young_professional_agent]

for idx, message in enumerate(st.session_state.chat_history): 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"])
        
        # Display extracted parameters if available
        if "params" in message and message["params"]:
            with st.expander("🔍 Extracted Search Parameters"):
                st.json(message["params"])
        
        if message["role"] == "assistant":
            col1, col2, col3 = st.columns([1, 1, 10])
            
            user_prompt = get_last_user_prompt(st.session_state.chat_history, idx)
            
            with col1:
                st.button(
                    "👍", 
                    key=f"like_{idx}",
                    on_click=handle_feedback,
                    args=(idx, user_prompt, message["text"], "like")
                )
                    
            with col2:
                st.button(
                    "👎", 
                    key=f"dislike_{idx}",
                    on_click=handle_feedback,
                    args=(idx, user_prompt, message["text"], "dislike")
                )
            
            if idx in st.session_state.feedback:
                with col3:
                    feedback_icon = "👍" if st.session_state.feedback[idx] == "like" else "👎"
                    st.caption(f"Feedback recorded: {feedback_icon}")

user_input = st.chat_input("Ask me anything...")
if user_input: 
    with st.chat_message("user"): 
        st.markdown(user_input) 

    st.session_state.chat_history.append({
        "role": "user",
        "text": user_input
    })

    question = user_input

    with st.spinner("🔍 Analyzing your query..."):
        search_params = extractor.extract(question)

    print(f"DEBUG: Extracted Params: {search_params}")

    if search_params:
        with st.chat_message("assistant"):
            st.info("📋 **I found these search parameters:**")
            st.markdown(search_params)

        st.session_state.chat_history.append({
            "role": "assistant",
            "text": f"📋 **I found these search parameters:**\n{search_params}",
            "params": search_params
        })

    with st.spinner("🤖 Selecting the right agents..."):
        _, router_response = router.select_models(question)

    print(f'Router selected: {router_response}')

    selected_agents = []
    for model in router_response:
        for agent in agents:
            if model == agent.agent_type:
                selected_agents.append(agent)

    if not selected_agents:
        st.warning("No agents were selected. Using all agents.")
        selected_agents = agents

    responses = []

    for agent in selected_agents:
        with st.spinner(f"💭 {agent.agent_type.replace('_', ' ').title()} is thinking..."):
            print(f"DEBUG: Question: {question}")
            print(f"DEBUG: search_params: {search_params}")
            ai_response = agent.infer(question, search_params)

        responses.append((agent.agent_type, ai_response))

        with st.chat_message("assistant"): 
            agent_name = agent.agent_type.replace('_', ' ').title()
            st.markdown(f"**[{agent_name}]:**")
            st.markdown(ai_response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "text": f"**[{agent_name}]:**\n{ai_response}"
        })

    if len(selected_agents) > 1:
        with st.spinner("📝 Creating summary..."):
            summarization = summarizer.infer(str(responses))

        with st.chat_message("assistant"): 
            st.markdown("---")
            st.markdown(f"**[Summary]:**")
            st.markdown(summarization)

        st.session_state.chat_history.append({
            "role": "assistant",
            "text": f"**[Summary]:**\n{summarization}"
        })
    
    st.rerun()
