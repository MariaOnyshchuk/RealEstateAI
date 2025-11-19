from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, Optional
import json
from langchain_community.utilities import SQLDatabase

from typing import Literal

from langgraph.prebuilt import ToolNode
from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from agent import Agent

class ParameterExtractor:
    """Extracts structured search parameters from natural language queries"""

    def __init__(self, llm):
        self.llm = llm
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a parameter extraction system for real estate searches.
Extract the following information from the user's query and return as JSON: {{"location": "city or neighborhood name", "price": numeric value or null, "bedrooms": number or null, "property_type": "house/apartment/condo/townhouse or null", "keywords": ["other", "relevant", "search", "terms"]}}
Only include fields that are explicitly mentioned or strongly implied.
Return ONLY valid JSON, no additional text."""),
            ("user", "{query}")
        ])

    def extract(self, query: str) -> Dict[str, Any]:
        """Extract search parameters from natural language query"""
        print("********PROMPT********",self.extraction_prompt)
        chain = self.extraction_prompt | self.llm
        response = chain.invoke({"query": {query}})
        content = getattr(response, "content", str(response)).strip()
        content = content.replace("```json", "").replace("```", "")

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {}

        print("Extracted JSON:", result)
        print("Type:", type(result))
        return result

# class BaseAgent:
#     def __init__(self, llm, agent_type):
#         self.llm = llm
#         self.agent_type = agent_type

#         self.system_prompt = "You are a helpful real estate assistant." # Default

#     def set_prompt(self, prompt_text):
#         self.system_prompt = prompt_text

#     def infer(self, question, search_params=None):
#         params_str = ""

#         params_str = str(search_params) if search_params else "None"
#         context_str = self.system_prompt + "\nContext - Search Parameters found: {param_context}"

#         prompt = ChatPromptTemplate.from_messages([
#             ("system", self.system_prompt + "\nContext - Search Parameters found: {param_context}"),
#             ("user", "{query}")
#         ])
#         print('DEBUG: Context', context_str)
#         chain = prompt | self.llm

#         response = chain.invoke({
#             "query": question,
#             "param_context": params_str  # LangChain will safely insert the string here
#         })

#         if hasattr(response, 'content'):
#             return response.content
#         return str(response)
class BaseAgent:
    def __init__(self, llm, agent_type):
        self.llm = llm
        self.agent_type = agent_type

        # the SQL-powered agent (automatically inserts property_url)
        self.sql_agent = Agent(llm, agent_type)

        self.system_prompt = "You are a helpful real estate assistant."

    def set_prompt(self, prompt_text):
        self.system_prompt = prompt_text

    def infer(self, question: str, search_params=None):
        """
        This version NO LONGER calls the LLM directly.
        All logic is routed through the SQL agent.
        """

        # Convert parameters to text for prompt
        param_context = str(search_params) if search_params else "None"

        # Add system + context to user message
        enriched_question = (
            self.system_prompt +
            f"\n\nContext: {param_context}\n\n" +
            question
        )

        # Delegate to SQL agent
        answer = self.sql_agent.infer(enriched_question)

        return answer


# class FamilyAgent(BaseAgent):
#     def __init__(self, llm):
#         super().__init__(llm, 'family')
#         self.set_prompt("""
#             You are a Real Estate Agent specializing in Family needs.
#             Focus on: Schools, safety, parks, quiet neighborhoods, backyard space, and community events.
#             Tone: Warm, reassuring, and detailed about long-term living.
#             When answering, prioritize the well-being of children and stability.
#         """)


# class InvestorAgent(BaseAgent):
#     def __init__(self, llm):
#         super().__init__(llm, 'investor')
#         self.set_prompt("""
#             You are a Real Estate Investment Analyst.
#             Focus on: ROI (Return on Investment), Cap Rates, rental yields, up-and-coming neighborhoods, and resale value.
#             Tone: Professional, analytical, concise, and numbers-driven.
#             Do not care about 'cozy' feelings; care about the profit margin.
#         """)


# class YoungProfessionalAgent(BaseAgent):
#     def __init__(self, llm):
#         super().__init__(llm, 'young_professional')
#         self.set_prompt("""
#             You are a Real Estate Agent for Young Professionals and Digital Nomads.
#             Focus on: Proximity to nightlife/city center, high-speed internet, modern amenities (gyms, pools),
#             coffee shops, and easy commute/public transport.
#             Tone: Energetic, trendy, and fast-paced.
#         """)

class FamilyAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, 'family')
        self.set_prompt("""
            You are a Real Estate Agent specializing in Family needs.
            Focus on: Schools, safety, parks, quiet neighborhoods, backyard space, and community events.
            Tone: Warm and reassuring.
        """)


class InvestorAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, 'investor')
        self.set_prompt("""
            You are a Real Estate Investment Analyst.
            Focus on ROI, cap rates, rental income, up-and-coming areas.
            Tone: Analytical and concise.
        """)


class YoungProfessionalAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, 'young_professional')
        self.set_prompt("""
            You are a Real Estate Agent for Young Professionals.
            Focus on nightlife, coworking, amenities, transit access.
            Tone: Energetic and modern.
        """)
