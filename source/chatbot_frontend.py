import streamlit as st 
import chatbot_backend as backend
from agent import Agent
from router import Router
from summarizer import Summarizer

st.title("Real Estate Chatbot 🤖")

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

family_agent = Agent(backend._bedrock_llm, 'family')
investor_agent = Agent(backend._bedrock_llm, 'investor')
young_professional_agent = Agent(backend._bedrock_llm, 'young_professional')

agents = [family_agent, investor_agent, young_professional_agent]

# Display chat history
for message in st.session_state.chat_history: 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"]) 

# Chat input
user_input = st.chat_input("Ask me anything...")
if user_input: 
    # Display user message
    with st.chat_message("user"): 
        st.markdown(user_input) 
    st.session_state.chat_history.append({"role":"user", "text":user_input}) 

    question = user_input


    output, router_response = router.select_models(question)
    print(f'{router_response=}')
    selected_agents = []

    for model in router_response:
        for agent in agents:
            if model == agent.agent_type:
                selected_agents.append(agent)

    with st.chat_message("assistant"): 
        st.markdown(output)

    # Get AI response
    # ai_response = backend.get_ai_response(user_input, st.session_state.store)
    # ai_family_response = family_agent.infer(question)
    responses = []

    for agent in selected_agents:

        ai_response = agent.infer(question)
        responses.append((agent.agent_type, ai_response))
                                                
        with st.chat_message("assistant"): 
            st.markdown(f"[{agent.agent_type}]:\n" + ai_response)

    if len(selected_agents) > 1:
        summarization = summarizer.infer(str(responses))
        with st.chat_message("assistant"): 
            st.markdown(f"[Summary]:\n" + summarization)
    
        # st.markdown(answers[-1].content)
    # st.session_state.chat_history.append({"role":"assistant", "text":ai_response}) 