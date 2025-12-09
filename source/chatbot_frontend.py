import streamlit as st 
import pandas as pd
import chatbot_backend as backend
from router import Router
from summarizer import Summarizer
from agents.base_agent import ParameterExtractor, AgentResponse, SpecializedAgent
from property_retrieval import Property
import json
import os
import logging
logging.basicConfig(level=logging.INFO)


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
        "Address", "Price", "Beds", "Baths", "Size",
        "Schools", "Parks",
        "Pharmacies", "Supermarkets",
        "Crime rate", "URL"
    ],
    "investor": [
        "Address", "Price", "ROI", "Cap Rate", "Annual Rent",
        "Vacancy Rate", "Property Taxes", "Appreciation", "URL"
    ],
    "young_professional": [
        "Address", "Price", "Beds", "Baths",
        "Commute Time", "Nightlife Score",
        "Coworking Spaces Nearby", "URL"
    ]
}


def format_property_value(prop, col):
    """Convert a Property field into a displayed UI value."""

    # basic fields
    if col == "Address":
        return prop.full_street_line or "N/A"
    if col == "Price":
        return f"${prop.list_price:,.0f}" if prop.list_price else "N/A"
    if col == "Beds":
        return prop.beds or "N/A"
    if col == "Baths":
        return prop.full_baths or "N/A"
    if col == "Size":
        return f"{prop.sqft:,.0f} sqft" if prop.sqft else "N/A"
    if col == "URL":
        return prop.property_url or "N/A"

    # investor fields
    if col == "ROI":
        return f"{prop.roi:.1f}%" if getattr(prop, "roi", None) else "N/A"
    if col == "Cap Rate":
        return f"{prop.cap_rate:.2f}%" if getattr(prop, "cap_rate", None) else "N/A"
    if col == "Annual Rent":
        return f"${prop.annual_rent:,.0f}" if getattr(prop, "annual_rent", None) else "N/A"
    if col == "Vacancy Rate":
        return f"{prop.vacancy_rate:.1f}%" if getattr(prop, "vacancy_rate", None) else "N/A"
    if col == "Property Taxes":
        return f"${prop.tax_cost:,.0f}" if getattr(prop, "tax_cost", None) else "N/A"
    if col == "Appreciation":
        return f"{prop.appreciation:.1f}%" if getattr(prop, "appreciation", None) else "N/A"

    # family fields
    # if col == "Schools Nearby":
    #     if getattr(prop, "school", None) is not None:
    #         val = getattr(prop, "school")
    #         return str(val)  # just show the number
    #     return "N/A"
    if col == "Schools":
        return str(getattr(prop, "school", "N/A")) if getattr(prop, "school", None) is not None else "No schools"
    if col == "Parks":
        return str(getattr(prop, "park", "N/A")) if getattr(prop, "park", None) is not None else "No parks"
    if col == "Pharmacies":
        return str(getattr(prop, "pharmacy", "N/A")) if getattr(prop, "pharmacy", None) is not None else "No pharmacies"
    if col == "Supermarkets":
        return str(getattr(prop, "supermarket", "N/A")) if getattr(prop, "supermarket", None) is not None else "No supermarkets"
    if col == "Crime rate":
        return str(getattr(prop, "crime_rate", "N/A")) if getattr(prop, "crime_rate", None) is not None else "No data"

    # if col == "Commute Time":
    #     return f"{prop.commute_minutes} min" if getattr(prop, "commute_minutes", None) else "N/A"
    # if col == "Nightlife Score":
    #     return f"{prop.nightlife_score}/10" if getattr(prop, "nightlife_score", None) else "N/A"
    # if col == "Coworking Spaces Nearby":
    #     return "; ".join(prop.coworking_spaces or []) if getattr(prop, "coworking_spaces", None) else "N/A"

    return "N/A"

def show_agent_output(agent_response: AgentResponse, agent_name: str = ""):
    """Display both agent summary and properties in a unified way."""
    # if agent_name:
    #     st.markdown(f"### {agent_name}")
    print(f'agent_response: {type(agent_response)}{agent_response}')
    # if agent_response.summary:
    #     st.markdown(agent_response.summary)

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
        synthesis_prompt = f"""You are creating a unified, comprehensive answer for a real estate query.
Original Question: "{question}"
We consulted three specialized real estate agents (Family, Investor, Young Professional) and received these perspectives:
{json.dumps(agent_summaries, indent=2)}
We found {len(unique_properties)} total unique properties across all agents.

Your task: Create ONE cohesive summary that:
1. Explains that we analyzed the query from multiple expert perspectives
2. Synthesizes the key insights from all three agents into a unified narrative
3. Highlights what makes the recommended properties suitable
4. Addresses different aspects (family-friendliness, investment potential, lifestyle fit)
5. Provides actionable guidance without repeating information
6. Keeps it concise (3-4 paragraphs maximum)

Important: Write as if you're a single expert who considered all angles, NOT as separate agent responses.
Do not use phrases like "The Family Agent said..." - instead synthesize into unified insights.
Return ONLY the summary text, no JSON or formatting."""

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
    st.session_state.completed_agents = [] #
    
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
            except Exception:
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
            except Exception:
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

    selected_agents = []
    for model_type in router_response:
        if model_type in agents:
            selected_agents.append((model_type, agents[model_type]))
    if not selected_agents:
        selected_agents = [(k, v) for k, v in agents.items()]

    if is_ambiguous_query:
        if "ambiguous_finished" not in st.session_state.completed_agents:
            with st.spinner("🎯 Creating comprehensive answer..."):
                temp_responses = []
                for agent_type, agent in selected_agents:
                    try:
                        raw = agent.infer(question, search_params)
                        props = [Property(**p) for p in raw.get("top_properties", [])]
                        temp_responses.append((agent_type, AgentResponse(summary=raw.get("summary",""), top_properties=props)))
                    except Exception:
                        continue
                
                unified = create_unified_response(question, temp_responses, backend.get_bedrock_client())
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": f"### 🎯 Comprehensive Analysis\n\n{unified.summary}",
                    "properties": unified.top_properties,
                    "allow_feedback": True
                })
            
            st.session_state.completed_agents.append("ambiguous_finished")
            st.session_state.processing_active = False 
            st.rerun()
            
    else:
        all_done = True
        for agent_type, agent in selected_agents:
            if agent_type not in st.session_state.completed_agents:
                all_done = False
                agent_name = agent_type.replace('_', ' ').title() + " Agent"
                
                with st.spinner(f"💭 {agent_name} is analyzing properties..."):
                    try:
                        raw = agent.infer(question, search_params)
                        props = [Property(**p) for p in raw.get("top_properties", [])]
                        ai_resp = AgentResponse(summary=raw.get("summary",""), top_properties=props)
                    except Exception as e:
                        ai_resp = AgentResponse(summary=f"Error: {e}", top_properties=[])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "text": f"**{agent_name}**\n\n{ai_resp.summary}",
                        "properties": ai_resp.top_properties,
                        "agent_type": agent_type,
                        "allow_feedback": True
                    })
                
                st.session_state.completed_agents.append(agent_type)
                st.rerun() 

        if all_done:
            st.session_state.processing_active = False
            st.rerun()
