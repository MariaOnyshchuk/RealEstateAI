import streamlit as st 
import chatbot_backend as backend
from agent import Agent
from router import Router
from summarizer import Summarizer

st.title("Real Estate Chatbot 🤖")

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

family_agent = Agent(backend._bedrock_llm, 'family')
investor_agent = Agent(backend._bedrock_llm, 'investor')
young_professional_agent = Agent(backend._bedrock_llm, 'young_professional')

agents = [family_agent, investor_agent, young_professional_agent]

for idx, message in enumerate(st.session_state.chat_history): 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"])
        
        if message["role"] == "assistant":
            col1, col2, col3 = st.columns([1, 1, 10])
            
            with col1:
                if st.button("👍", key=f"like_{idx}"):
                    st.session_state.feedback[idx] = "like"
                    st.rerun()  
                    
            with col2:
                if st.button("👎", key=f"dislike_{idx}"):
                    st.session_state.feedback[idx] = "dislike"
                    st.rerun()  
            
            if idx in st.session_state.feedback:
                with col3:
                    feedback_icon = "👍" if st.session_state.feedback[idx] == "like" else "👎"
                    st.caption(f"Your feedback: {feedback_icon}")

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
    st.session_state.chat_history.append({"role":"assistant", "text":output})  

    responses = []

    for agent in selected_agents:
        ai_response = agent.infer(question)
        responses.append((agent.agent_type, ai_response))
        
        agent_text = f"[{agent.agent_type}]:\n{ai_response}" 
        with st.chat_message("assistant"): 
            st.markdown(agent_text)
        st.session_state.chat_history.append({"role":"assistant", "text":agent_text}) 

    if len(selected_agents) > 1:
        summarization = summarizer.infer(str(responses))
        summary_text = f"[Summary]:\n{summarization}"
        with st.chat_message("assistant"): 
            st.markdown(summary_text)
        st.session_state.chat_history.append({"role":"assistant", "text":summary_text})  
    
    st.rerun()  

with st.sidebar:
    if st.session_state.feedback:
        import json
        feedback_data = {
            "chat_history": st.session_state.chat_history,
            "feedback": st.session_state.feedback
        }
        st.download_button(
            "Download Feedback",
            data=json.dumps(feedback_data, indent=2, ensure_ascii=False),
            file_name="chatbot_feedback.json",
            mime="application/json"
        )
