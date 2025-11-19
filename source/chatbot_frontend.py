import streamlit as st 
import chatbot_backend as backend
from agent import Agent
from router import Router
from summarizer import Summarizer
from agents.base_agent import ParameterExtractor

st.title("Real Estate Chatbot 🏠")

# print(st.session_state)
if 'store' not in st.session_state:
    st.session_state.store = {}

# Initialize chat memory
if 'chat_memory' not in st.session_state: 
    st.session_state.chat_memory = backend.create_chat_memory()

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

router = Router(backend._bedrock_llm)
summarizer = Summarizer(backend._bedrock_llm)
extractor = ParameterExtractor(backend._bedrock_llm)

from agents.base_agent import FamilyAgent, InvestorAgent, YoungProfessionalAgent

family_agent = FamilyAgent(backend._bedrock_llm)
investor_agent = InvestorAgent(backend._bedrock_llm)
young_professional_agent = YoungProfessionalAgent(backend._bedrock_llm)

agents = [family_agent, investor_agent, young_professional_agent]

for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"]) 
        # Show extracted params if they exist
        if "params" in message and message["params"]:
            with st.expander("🔍 Extracted Search Parameters"):
                st.json(message["params"])

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
            print(f"DEBUG: Extracted Question: {question}")
            print(f"DEBUG: Extracted search_params: {search_params}")
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

    # --- STEP 4: Summarize (if multiple agents responded) ---
    if len(selected_agents) > 1:
        with st.spinner("📝 Creating summary..."):
            summarization = summarizer.infer(str(responses))

        with st.chat_message("assistant"): 
            st.markdown("---")
            st.markdown(f"**[Summary]:**")
            st.markdown(summarization)

        # Store summary in chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "text": f"**[Summary]:**\n{summarization}"
        })
