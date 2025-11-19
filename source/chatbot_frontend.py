import streamlit as st 
import chatbot_backend as backend
from agent import Agent

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

family_agent = Agent(backend._bedrock_llm, 'family')
investor_agent = Agent(backend._bedrock_llm, 'investor')
young_professional_agent = Agent(backend._bedrock_llm, 'young_professional')

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


    # Get AI response
    # ai_response = backend.get_ai_response(user_input, st.session_state.store)
    ai_family_response = family_agent.infer(question)
    with st.chat_message("assistant"): 
        st.markdown(ai_family_response)

    ai_investor_response = investor_agent.infer(question)
    with st.chat_message("assistant"): 
        st.markdown(ai_investor_response)

    ai_young_professional_response = young_professional_agent.infer(question)
    with st.chat_message("assistant"): 
        st.markdown(ai_young_professional_response)
    
        # st.markdown(answers[-1].content)
    # st.session_state.chat_history.append({"role":"assistant", "text":ai_response}) 