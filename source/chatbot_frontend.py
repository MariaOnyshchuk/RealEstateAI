import streamlit as st 
import pandas as pd
import chatbot_backend as backend
from router import Router
from summarizer import Summarizer
from agents.base_agent import ParameterExtractor, AgentResponse, SpecializedAgent
from agents.prompts import UNIFIED_ANSWER_PROMPT
from property_retrieval import Property
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed #paralell

logging.basicConfig(level=logging.INFO)

def process_single_agent(agent_type, agent, question, search_params):
    """Process a single agent and return results"""
    agent_name = agent_type.replace('_', ' ').title() + " Agent"

    try:
        logging.debug(f"CLASS OF AGENT: {type(agent)}")
        raw_response = agent.infer(question, search_params)
        logging.debug(f'after infer: {raw_response}')

        properties = []
        for p in raw_response.get("top_properties", []):
            prop_obj = Property(**p)
            properties.append(prop_obj)

        ai_response = AgentResponse(
            summary=raw_response.get("summary", ""),
            top_properties=properties
        )
        return (agent_type, ai_response, None)  # (type, response, error)

    except Exception as e:
        logging.error(f"Error from {agent_name}: {e}")
        ai_response = AgentResponse(
            summary=f"Error processing request: {str(e)}",
            top_properties=[]
        )
        return (agent_type, ai_response, str(e))

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

def handle_streamlit_feedback(idx, user_prompt, ai_text):
    key = f"fb_{idx}"
    score = st.session_state.get(key)
    
    if score == 1:
        st.session_state.feedback[idx] = "like"
        save_feedback_for_kto(user_prompt, ai_text, True)
        
    elif score == 0:
        st.session_state.feedback[idx] = "dislike"
        save_feedback_for_kto(user_prompt, ai_text, False)
        
    else:
        if idx in st.session_state.feedback:
            del st.session_state.feedback[idx]

st.set_page_config(
    page_title="Real Estate Chatbot",
    page_icon="🏠",
    layout="wide"
)

st.title("Real Estate Chatbot 🏠")

if 'store' not in st.session_state:
    st.session_state.store = {}

if 'chat_memory' not in st.session_state: 
    st.session_state.chat_memory = backend.create_chat_memory()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

llm_client = backend.get_bedrock_client()

router = Router(llm_client)
summarizer = Summarizer(llm_client)
extractor = ParameterExtractor(llm_client)

agents = {
    agent_type: SpecializedAgent(llm_client, agent_type)
    for agent_type in ['family', 'investor', 'young_professional']
}

def format_params_display(params: dict) -> str:
    """Format extracted parameters for nice display"""
    if not params:
        return "No specific parameters extracted"

    display_parts = []

    if params.get("location"):
        display_parts.append(f"📍 **Location:** {params['location']}")

    price_parts = []
    if params.get("price_min"):
        price_parts.append(f"${params['price_min']:,}")
    if params.get("price_max"):
        if price_parts:
            price_parts.append(f"${params['price_max']:,}")
        else:
            price_parts.append(f"up to ${params['price_max']:,}")
    if price_parts:
        display_parts.append(f"💰 **Price:** {' - '.join(price_parts)}")

    if params.get("bedrooms_min") or params.get("bedrooms_max"):
        bed_str = ""
        if params.get("bedrooms_min") and params.get("bedrooms_max"):
            bed_str = f"{params['bedrooms_min']}-{params['bedrooms_max']}"
        elif params.get("bedrooms_min"):
            bed_str = f"{params['bedrooms_min']}+"
        else:
            bed_str = f"up to {params['bedrooms_max']}"
        display_parts.append(f"🛏️ **Bedrooms:** {bed_str}")

    if params.get("bathrooms_min") or params.get("bathrooms_max"):
        bath_str = ""
        if params.get("bathrooms_min") and params.get("bathrooms_max"):
            bath_str = f"{params['bathrooms_min']}-{params['bathrooms_max']}"
        elif params.get("bathrooms_min"):
            bath_str = f"{params['bathrooms_min']}+"
        else:
            bath_str = f"up to {params['bathrooms_max']}"
        display_parts.append(f"🚿 **Bathrooms:** {bath_str}")

    if params.get("size_min") or params.get("size_max"):
        size_str = ""
        if params.get("size_min") and params.get("size_max"):
            size_str = f"{params['size_min']:,}-{params['size_max']:,} sqft"
        elif params.get("size_min"):
            size_str = f"{params['size_min']:,}+ sqft"
        else:
            size_str = f"up to {params['size_max']:,} sqft"
        display_parts.append(f"📏 **Size:** {size_str}")

    if params.get("property_type"):
        display_parts.append(f"🏘️ **Type:** {params['property_type'].title()}")

    if params.get("amenities"):
        amenities_str = ", ".join(params['amenities'])
        display_parts.append(f"✨ **Amenities:** {amenities_str}")

    if params.get("preferences"):
        prefs_str = ", ".join(params['preferences'])
        display_parts.append(f"💭 **Preferences:** {prefs_str}")

    if params.get("lifestyle"):
        display_parts.append(f"👤 **Lifestyle:** {params['lifestyle'].replace('_', ' ').title()}")

    return "\n".join(display_parts)


AGENT_COLUMN_CONFIG = {
    "family": [
        "URL", "Address", "Price", "Beds", "Baths", "Size", "Stories",
        "Schools",
        "Pharmacies", "Supermarkets",
        "Crime rate"
        #   "Shopping Mall", "Parks",
    ],
    "investor": [
        "URL" , "Address", "Price", "ROI", "Monthly Rent", "Stories",
        "Property Taxes"
        # "Maintenance", "HOA Fee"
    ],
    "young_professional": [
        "URL", "Address", "Price", "Beds", "Baths",
        "Restaurants", "Gyms", "Stories",
        "Town Square", "Libraries", "Night Clubs"
    ],
    'comprehensive': [
        "URL", "Address", "Price", "Beds", "Baths", "Size", "Stories",
        "Schools",
        "Property Taxes", "Restaurants",  "Gyms"
    ]
}


def format_property_value(prop, col):
    """Convert a Property field into a displayed UI value."""

    # basic fields
    if col == "URL":
        return prop.property_url or 'No data'
    if col == "Address":
        return prop.full_street_line or 'No data'
    if col == "Price":
        return f"${prop.list_price:,.0f}" if prop.list_price else 'No data'
    if col == "Beds":
        return prop.beds or 'No data'
    if col == "Baths":
        return prop.full_baths or 'No data'
    if col == "Size":
        return f"{prop.sqft:,.0f} sqft" if prop.sqft else 'No data'
    if col == "Stories":
        return prop.stories or 'No data'

    # investor fields
    if col == "ROI":
        # return str(getattr(prop, "roi", "No data")) if getattr(prop, "roi", None) is not None else "No data"
        return f"{prop.roi:,.2f}%" if getattr(prop, "roi", None) else 'No data'
    if col == "Monthly Rent":
        return f"${prop.avg_monthly_rent:,.0f}" if getattr(prop, "avg_monthly_rent", None) else 'No data'
    if col == "Property Taxes":
        return f"${prop.tax:,.0f}" if getattr(prop, "tax", None) else 'No data'
    # if col == "Maintenance":
    #     return f"${prop.maintenance:,.0f}" if getattr(prop, "maintenance", None) else 'No data'
    # if col =="HOA Fee":
    #     return f"${prop.hoa_fee:,.0f}" if getattr(prop, "hoa_fee", None) else 'No data'

    # family fields
    if col == "Schools":
        return str(getattr(prop, "school", 'No data')) if getattr(prop, "school", None) is not None else "No schools"
    if col == "Parks":
        return str(getattr(prop, "park", 'No data')) if getattr(prop, "park", None) is not None else "No parks"
    if col == "Pharmacies":
        return str(getattr(prop, "pharmacy", 'No data')) if getattr(prop, "pharmacy", None) is not None else "No pharmacies"
    if col == "Supermarkets":
        return str(getattr(prop, "supermarket", 'No data')) if getattr(prop, "supermarket", None) is not None else "No supermarkets"
    if col == "Crime rate":
        return str(getattr(prop, "crime_rate", 'No data')) if getattr(prop, "crime_rate", None) is not None else "No data"
    if col == "Shopping Mall":
        return prop.shopping_mall or 'No data'

    if col == "Restaurants":
        return str(getattr(prop, "restaurant", 'No data')) if getattr(prop, "restaurant", None) is not None else "No data"
    if col == "Gyms":
        return prop.gym or 'No data'
    if col =="Libraries":
        return prop.library or 'No data'
    if col == "Night Clubs":
        return str(getattr(prop, "night_club", 'No data')) if getattr(prop, "night_club", None) is not None else "No data"
    if col == "Town Square":
        return prop.town_square or 'No data'

    return 'Need to be added to list of propetries'

def show_agent_output(agent_response: AgentResponse, agent_name: str = ""):
    print(f'agent_response: {type(agent_response)}{agent_response}')
    if agent_response.top_properties:
        st.markdown("#### 🏠 Recommended Properties")
        columns = AGENT_COLUMN_CONFIG.get(agent_name.lower(), AGENT_COLUMN_CONFIG["family"])
        df_data = []
        for prop in agent_response.top_properties:
            row = {col: format_property_value(prop, col) for col in columns}
            df_data.append(row)
        df = pd.DataFrame(df_data)

        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            column_config={"URL": st.column_config.LinkColumn("Property Link")}
        )
    else:
        st.info("No properties found for this search.")

def create_unified_response(question: str, agent_responses: list, llm) -> AgentResponse:

    try:
        all_properties = []
        agent_summaries = []

        for agent_type, response in agent_responses:
            agent_name = agent_type.replace('_', ' ').title()
            agent_summaries.append({
                "agent": agent_name,
                "perspective": response.summary,
                "property_count": len(response.top_properties)
            })

            for prop in response.top_properties:
                all_properties.append({
                    "agent_source": agent_name,
                    "property": prop
                })

        seen_addresses = set()
        unique_properties = []
        for item in all_properties:
            prop = item["property"]
            address_key = (prop.full_street_line or prop.property_url)

            if address_key not in seen_addresses:
                seen_addresses.add(address_key)
                unique_properties.append(prop)
        print('UNIQUE PROPS:', len(unique_properties))

        def score_property(prop):
            score = 0
            if prop.roi:
                score += prop.roi * 10
            if prop.beds:
                score += prop.beds * 2
            if prop.list_price:
                if prop.list_price < 300_000:
                    score += 5
                elif prop.list_price < 500_000:
                    score += 3
            return score

        unique_properties.sort(key=score_property, reverse=True)
        top_properties = unique_properties

        synthesis_prompt = UNIFIED_ANSWER_PROMPT.format(
            question=question,
            all_summaries={json.dumps(agent_summaries, indent=2)},
            unique_properties_num=len(unique_properties)
        )

        unified_summary = llm.invoke([
            {"role": "system", "content": "You are a comprehensive real estate advisor."},
            {"role": "user", "content": synthesis_prompt}
        ]).content

        return AgentResponse(
            summary=unified_summary.strip(),
            top_properties=top_properties
        )

    except Exception as e:
        print(f"Error creating unified response: {e}")
        combined_summary = f"Based on comprehensive analysis from multiple expert perspectives:\n\n"

        for agent_type, response in agent_responses:
            agent_name = agent_type.replace('_', ' ').title()
            combined_summary += f"**{agent_name} Perspective:** {response.summary}\n\n"

        fallback_properties = agent_responses[0][1].top_properties if agent_responses else []

        return AgentResponse(
            summary=combined_summary,
            top_properties=fallback_properties
        )


for idx, message in enumerate(st.session_state.chat_history): 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"])
        
        if message["role"] == "assistant" and "params" in message and message["params"]:
            with st.expander("🔍 Extracted Search Parameters"):
                st.markdown(format_params_display(message["params"]))
        
        if message["role"] == "assistant" and "properties" in message and message["properties"]:
            show_agent_output(
                AgentResponse(
                    summary=message.get("text", ""),
                    top_properties=message["properties"]
                ),
                agent_name=message.get("agent_type", "")
            )

        if message["role"] == "assistant" and message.get("allow_feedback", False):
            user_prompt = get_last_user_prompt(st.session_state.chat_history, idx)
            
            existing_vote = st.session_state.feedback.get(idx)
            widget_key = f"fb_{idx}"
            
            if widget_key not in st.session_state:
                if existing_vote == "like":
                    st.session_state[widget_key] = 1
                elif existing_vote == "dislike":
                    st.session_state[widget_key] = 0
                else:
                    st.session_state[widget_key] = None
            
            st.feedback(
                "thumbs", 
                key=widget_key,
                on_change=handle_streamlit_feedback,
                args=(idx, user_prompt, message["text"])
            )
            

if 'processing_active' not in st.session_state:
    st.session_state.processing_active = False
if 'current_query' not in st.session_state:
    st.session_state.current_query = None
if 'completed_agents' not in st.session_state:
    st.session_state.completed_agents = []


user_input = st.chat_input("Ask me anything about real estate...")

if user_input:
    st.session_state.current_query = user_input
    st.session_state.processing_active = True
    st.session_state.completed_agents = []

    st.session_state.chat_history.append({
        "role": "user",
        "text": user_input
    })
    st.rerun()

if st.session_state.processing_active:
    question = st.session_state.current_query

    if "params_extracted" not in st.session_state.completed_agents:
        with st.spinner("🔍 Analyzing your query..."):
            try:
                search_params = extractor.extract(question)
            except Exception as e:
                st.error(f"Error extracting parameters: {e}")
                search_params = {"raw_query": question}

            st.session_state.last_params = search_params

            if search_params and any(v for k, v in search_params.items() if k not in ['raw_query', 'keywords']):
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": "### 📋 Search Parameters\n" + format_params_display(search_params),
                    "params": search_params,
                    "allow_feedback": False
                })

        st.session_state.completed_agents.append("params_extracted")
        st.rerun()

    search_params = st.session_state.get("last_params", {})

    if "routing_done" not in st.session_state.completed_agents:
        with st.spinner("🤖 Selecting agents..."):
            try:
                scores, router_response = router.select_models(question)
                is_ambiguous = (len(router_response) == len(agents))
            except Exception as e:
                st.warning(f"Router error: {e}. Using all agents.")
                router_response = ['family', 'investor', 'young_professional']
                is_ambiguous = True

            st.session_state.last_router_response = router_response
            st.session_state.last_is_ambiguous = is_ambiguous

            if is_ambiguous:
                st.info("🔀 Ambiguous query - consulting all agents")
            else:
                names = [n.replace('_', ' ').title() for n in router_response]
                st.info(f"🎯 Routing to: {', '.join(names)}")

        st.session_state.completed_agents.append("routing_done")
        st.rerun()

    router_response = st.session_state.get("last_router_response", [])
    is_ambiguous_query = st.session_state.get("last_is_ambiguous", False)

    if "agents_executed" not in st.session_state.completed_agents:

        selected_agents = []
        for model_type in router_response:
            if model_type in agents:
                selected_agents.append((model_type, agents[model_type]))
        if not selected_agents:
            selected_agents = [(k, v) for k, v in agents.items()]

        if len(selected_agents) == 1:
            agent_name = selected_agents[0][0].replace('_', ' ').title() + " Agent"
            spinner_text = f"💭 {agent_name} is analyzing properties..."
        else:
            spinner_text = "💭 All agents are analyzing properties in parallel..."

        responses_buffer = []

        with st.spinner(spinner_text):
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_agent = {
                    executor.submit(process_single_agent, atype, agent, question, search_params): atype
                    for atype, agent in selected_agents
                }

                for future in as_completed(future_to_agent):
                    atype, ai_response, error = future.result()

                    if error:
                        st.error(f"Error from {atype}: {error}")
                        err_resp = AgentResponse(summary=f"Error: {error}", top_properties=[])
                        responses_buffer.append((atype, err_resp))
                    else:
                        responses_buffer.append((atype, ai_response))

        order_priority = ['family', 'investor', 'young_professional']
        responses_buffer.sort(key=lambda x: order_priority.index(x[0]) if x[0] in order_priority else 99)


        if is_ambiguous_query:
            with st.spinner("🎯 Synthesizing comprehensive answer..."):
                unified = create_unified_response(question, responses_buffer, llm_client)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": f"### 🎯 Comprehensive Analysis\n\n{unified.summary}",
                    "properties": unified.top_properties,
                    "allow_feedback": True
                })
        else:
            for agent_type, ai_response in responses_buffer:
                agent_name = agent_type.replace('_', ' ').title() + " Agent"

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": f"**{agent_name}**\n\n{ai_response.summary}",
                    "properties": ai_response.top_properties,
                    "agent_type": agent_type,
                    "allow_feedback": True
                })

        st.session_state.completed_agents.append("agents_executed")
        st.session_state.processing_active = False
        st.rerun()

with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🗑️ Clear Chat History", key="clear_chat_button"):
        st.session_state.chat_history = []
        st.session_state.feedback = {}
        st.rerun()

    if st.session_state.feedback:
        st.markdown("---")
        st.subheader("📊 Chat Statistics")
        likes = len([f for f in st.session_state.feedback.values() if f == "like"])
        dislikes = len([f for f in st.session_state.feedback.values() if f == "dislike"])
        st.metric("👍 Likes", likes)
        st.metric("👎 Dislikes", dislikes)

    if st.session_state.feedback:
        st.markdown("---")
        st.subheader("💾 Export Data")
        
        feedback_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "chat_history": st.session_state.chat_history,
            "feedback": st.session_state.feedback
        }

        st.download_button(
            label="📥 Download Feedback",
            data=json.dumps(feedback_data, indent=2, default=str),
            file_name=f"chatbot_feedback_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_feedback_button"
        )
    st.markdown("---")
    st.subheader("🚀 Quick Queries")
    st.caption("Click to fill the input field with sample queries")

    # Define your predefined queries
    quick_queries = {
        "👨‍👩‍👧‍👦 Family Home": "I'm looking for a family-friendly home with 3-4 bedrooms, 2 schools nearby, and a safe neighborhood",
        "💼 Investment Property": "Show me properties with high ROI potential for investment purposes",
        "🏃 Young Professional": "I need a modern apartment near restaurants, gyms, and nightlife",
        "🏠 Budget Friendly": "Find affordable homes under $300,000 with at least 2 bedrooms"
    }
    # Create buttons for each query
    for idx, (label, query) in enumerate(quick_queries.items()):
        if st.button(label, key=f"quick_query_{idx}", use_container_width=True):
            st.session_state.current_query = query
            st.session_state.processing_active = True
            st.session_state.completed_agents = []

            st.session_state.chat_history.append({
                "role": "user",
                "text": query
            })
            st.rerun()

    st.markdown("---")
