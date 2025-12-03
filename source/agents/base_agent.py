from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, Optional, List
import json
import re
from pydantic import BaseModel, ConfigDict, Field
from agents.prompts import EXTRACT_PARAMS_FROM_QUERY_PROMPT, FAMIILY_AGENT, INVESTOR_AGENT, YOUNG_PROFESSIONAL
from agent import Agent
from property_retrieval import Property


class AgentResponse(BaseModel):
    """Model for agent response with validation"""
    model_config = ConfigDict(extra="allow")

    summary: str = ""
    top_properties: List[Property] = Field(default_factory=list)


class ParameterExtractor:
    """Extract search parameters from natural language queries"""

    def __init__(self, llm):
        self.llm = llm
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACT_PARAMS_FROM_QUERY_PROMPT),
            ("user", "{query}")
        ])

    def extract(self, query: str) -> Dict[str, Any]:
        try:
            chain = self.extraction_prompt | self.llm
            response = chain.invoke({"query": query})
            content = getattr(response, "content", str(response)).strip()
            content = re.sub(r'```json\s*|\s*```', '', content, flags=re.IGNORECASE)
            result = json.loads(content)
            result["raw_query"] = query
            return result
        except Exception as e:
            print(f"Extraction error: {e}")
            return {"keywords": [query], "raw_query": query}


class BaseAgent:
    def __init__(self, llm, agent_type: str = "generic"):
        self.llm = llm
        self.agent_type = agent_type
        self.sql_agent = Agent(llm, agent_type)
        self.system_prompt = "You are a helpful real estate assistant."

    def set_prompt(self, prompt_text: str):
        self.system_prompt = prompt_text

    def infer(self, question: str, search_params: Optional[Dict] = None) -> str:
        context = ""
        if search_params:
            context = "\n".join([f"- {k}: {v}" for k, v in search_params.items() if v])
        prompt = f"{self.agent_type} agent context:\n{context}\n\nUser Question: {question}\nReturn valid JSON only."
        try:
            return self.sql_agent.infer(prompt)
        except Exception as e:
            print(f"Inference error: {e}")
            return json.dumps({"summary": str(e), "top_properties": []})


class SpecializedAgent(BaseAgent):
    PROMPTS = {
        "family": FAMIILY_AGENT,
        "investor": INVESTOR_AGENT,
        "young_professional": YOUNG_PROFESSIONAL
    }

    def __init__(self, llm, agent_type):
        super().__init__(llm, agent_type)
        if agent_type in self.PROMPTS:
            self.set_prompt(self.PROMPTS[agent_type])

    def infer(self, question, search_params=None):
        context = ""
        if search_params:
            context = "\n".join(f"{k}: {v}" for k, v in search_params.items() if v)

        prompt = (
            self.system_prompt
            + "\n\n"
            + "Context:\n"
            + context
            + "\n\nUser question: "
            + question
            + "\nReturn valid JSON only."
        )

        return self.sql_agent.infer(prompt)
