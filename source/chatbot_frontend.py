import streamlit as st 
import pandas as pd
import json
import chatbot_backend as backend
from router import Router
from summarizer import Summarizer
from agents.base_agent import ParameterExtractor, AgentResponse, SpecializedAgent, Property
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

router = Router(backend._bedrock_llm)
summarizer = Summarizer(backend._bedrock_llm)
extractor = ParameterExtractor(backend._bedrock_llm)


agents = {
    agent_type: SpecializedAgent(backend._bedrock_llm, agent_type)
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


def display_properties_table(properties, agent_name: str = ""):
    """Display properties in a nice Streamlit table"""
    if not properties:
        st.info("No properties found matching your criteria.")
        return

    df_data = []
    for prop in properties:
        df_data.append({
            "Address": prop.full_street_line or "N/A",
            "Price": f"${prop.list_price:,.0f}" if prop.list_price else "N/A",
            "Beds": prop.beds if prop.beds is not None else "N/A",
            "Baths": prop.full_baths if prop.full_baths is not None else "N/A",
            "Size": f"{prop.sqft:,.0f} sqft" if prop.sqft else "N/A",
            "Type": prop.style.title() if prop.style else "N/A",
            "ROI": f"{prop.roi:.1f}%" if prop.roi else "N/A",
            "URL": prop.property_url or "N/A"
        })

    df = pd.DataFrame(df_data)

    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "URL": st.column_config.LinkColumn("Property Link")
        }
    )


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


def display_agent_response(agent_response: AgentResponse, agent_name: str):
    st.markdown(f"### {agent_name}")

    if agent_response.summary:
        st.markdown(agent_response.summary)

    if agent_response.top_properties:
        st.markdown("#### 🏠 Recommended Properties")
        display_properties_table(agent_response.top_properties, agent_name)
    else:
        st.info("No properties found for this search.")


for idx, message in enumerate(st.session_state.chat_history): 
    with st.chat_message(message["role"]): 
        st.markdown(message["text"])
        
        if message["role"] == "assistant" and "params" in message and message["params"]:
            with st.expander("🔍 Extracted Search Parameters"):
                st.markdown(format_params_display(message["params"]))
        
        if message["role"] == "assistant" and "properties" in message and message["properties"]:
            display_properties_table(message["properties"])

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

user_input = st.chat_input("Ask me anything about real estate...",)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({
        "role": "user",
        "text": user_input
    })

    question = user_input

    with st.spinner("🔍 Analyzing your query..."):
        try:
            search_params = extractor.extract(question)
        except Exception as e:
            st.error(f"Error extracting parameters: {e}")
            search_params = {"raw_query": question}

    print(f"DEBUG: Extracted Params: {search_params}")

    if search_params and any(v for k, v in search_params.items() if k not in ['raw_query', 'keywords']):
        with st.chat_message("assistant"):
            st.markdown("### 📋 Search Parameters")
            st.markdown(format_params_display(search_params))

        st.session_state.chat_history.append({
            "role": "assistant",
            "text": "### 📋 Search Parameters\n" + format_params_display(search_params),
            "params": search_params,
            "allow_feedback": False
        })

    with st.spinner("🤖 Analyzing query and selecting agents..."):
        try:
            scores, router_response = router.select_models(question)
            print(f'Router scores: {scores}')
            print(f'Router selected: {router_response}')

            is_ambiguous_query = len(router_response) == len(agents)

            if is_ambiguous_query:
                st.info("🔀 Query is ambiguous - consulting all agents for unified comprehensive answer")
            else:
                selected_names = [name.replace('_', ' ').title() for name in router_response]
                st.info(f"🎯 Routing to: {', '.join(selected_names)}")

        except Exception as e:
            st.warning(f"Router error: {e}. Using all agents.")
            router_response = ['family', 'investor', 'young_professional']
            scores = {agent: 0.33 for agent in router_response}
            is_ambiguous_query = True

    # Get selected agents
    selected_agents = []
    for model_type in router_response:
        if model_type in agents:
            selected_agents.append((model_type, agents[model_type]))

    # Fallback to all agents if none selected
    if not selected_agents:
        st.info("No specific agents selected. Using all available agents.")
        selected_agents = [(k, v) for k, v in agents.items()]

    responses = []

    for agent_type, agent in selected_agents:
        agent_name = agent_type.replace('_', ' ').title() + " Agent"

        with st.spinner(f"💭 {agent_name} is analyzing properties..."):
            try:
                print('CLASS OF AGENT', type(agent))
                raw_response = agent.infer(question, search_params)
                print('after infer', raw_response)
                print('raw_response type', type(raw_response))
                # raw_response= json.loads(raw_response)
                print('converted', type(raw_response), raw_response)
                properties = []
                for p in raw_response.get("top_properties", []):
                    prop_obj = Property(
                        property_url=p.get("property_url", ""),
                        full_street_line=p.get("full_street_line"),
                        list_price=p.get("list_price"),
                        beds=p.get("beds"),
                        full_baths=p.get("full_baths"),
                        sqft=p.get("sqft"),
                        style=p.get("style"),
                        roi=p.get("roi")
                    )
                    properties.append(prop_obj)

                print(raw_response)
                ai_response = AgentResponse(
                    summary=raw_response.get("summary", ""),
                    top_properties=properties
                )

            except Exception as e:
                st.error(f"Error from {agent_name}: {e}")
                ai_response = AgentResponse(
                    summary=f"Error processing request: {str(e)}",
                    top_properties=[]
                )

        responses.append((agent_type, ai_response))


    if is_ambiguous_query:
        with st.spinner("🎯 Creating unified comprehensive answer..."):
            unified_response = create_unified_response(question, responses, backend._bedrock_llm)

        with st.chat_message("assistant"):
            st.markdown("### 🎯 Comprehensive Analysis")
            st.markdown(unified_response.summary)

            if unified_response.top_properties:
                st.markdown("#### 🏠 Top Recommended Properties")
                display_properties_table(unified_response.top_properties)

        st.session_state.chat_history.append({
            "role": "assistant",
            "text": f"### 🎯 Comprehensive Analysis\n\n{unified_response.summary}",
            "properties": unified_response.top_properties,
            "allow_feedback": True
        })
    else:

        for agent_type, ai_response in responses:
            agent_name = agent_type.replace('_', ' ').title() + " Agent"

            with st.chat_message("assistant"):
                display_agent_response(ai_response, agent_name)

            # Store in chat history
            response_text = f"**{agent_name}**\n\n{ai_response.summary}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "text": response_text,
                "properties": ai_response.top_properties,
                "agent_type": agent_type,
                "allow_feedback": True
            })

    if len(selected_agents) > 1 and not is_ambiguous_query:
        with st.spinner("📝 Creating comprehensive summary..."):
            try:
                is_ambiguous = len(selected_agents) == len(agents)
                summary_input = {
                    "query": question,
                    "is_ambiguous": is_ambiguous,
                    "agent_responses": []
                }



                # summary_input = []
                for agent_type, response in responses:
                    summary_input["agent_responses"].append({
                        "agent": agent_type.replace('_', ' ').title(),
                        "summary": response.summary,
                        "property_count": len(response.top_properties),
                        "sample_properties": [
                            {
                                "address": prop.full_street_line,
                                "price": prop.price,
                                "bedrooms": prop.bedrooms
                            }
                            for prop in response.top_properties[:2]  # Just top 2 for summary
                        ]
                    })

                summary_prompt = f"""Multiple specialized agents responded to: "{question}"

{json.dumps(summary_input["agent_responses"], indent=2, default=str)}

Create a concise summary that:
1. Synthesizes the key points from each agent's perspective
2. Highlights areas of agreement and difference
3. Provides overall guidance based on all perspectives
4. Keeps it brief (2-3 paragraphs)"""
                summarization = backend._bedrock_llm.invoke([
                    {"role": "system", "content": "You are a real estate assistant creating perspective summaries."},
                    {"role": "user", "content": summary_prompt}
                ]).content
            except Exception as e:
                st.warning(f"Could not generate summary: {e}")
                summarization = "Multiple specialized agents provided recommendations based on their expertise. Please review each agent's suggestions above."

        with st.chat_message("assistant"):
            st.markdown("---")
            st.markdown("### 📊 Summary of Perspectives")
            st.markdown(summarization)

        st.session_state.chat_history.append({
            "role": "assistant",
            "text": f"### 📊 Summary of Perspectives\n\n{summarization}",
            "allow_feedback": True
        })

    st.rerun()

with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🗑️ Clear Chat History"):
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
            mime="application/json"
        )

    st.markdown("---")
    st.subheader("💡 Tips")
    st.markdown("""
    **You can ask:**
    - Specific queries: *"3 bed house in Seattle under $500k"*
    - Abstract requests: *"cozy place near downtown"*
    - Lifestyle-based: *"good for families with kids"*
    - Investment focused: *"properties with high ROI"*

    The chatbot understands natural language!
    """)
