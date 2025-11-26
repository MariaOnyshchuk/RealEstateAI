from langchain.agents import create_agent
from typing import List, Dict, Tuple
import re


class Router:
    """
    Intelligent router that determines which real estate agent(s) should handle a query.
    Uses keyword matching and LLM-based analysis for routing decisions.
    """

    AVAILABLE_MODELS = ["family", "investor", "young_professional"]

    # Parameter-based routing indicators
    ROUTING_INDICATORS = {
        "family": {
            "keywords": [
                "family", "families", "kids", "children", "child", "school", "schools",
                "playground", "park", "safe", "safety", "quiet", "neighborhood",
                "backyard", "yard", "garden", "family-friendly", "elementary",
                "daycare", "suburban", "cul-de-sac", "community"
            ],
            "phrases": [
                "good schools", "raising kids", "raise a family", "school district",
                "safe neighborhood", "family home", "kids room", "play area"
            ],
            "anti_keywords": []  # Keywords that reduce family score
        },
        "investor": {
            "keywords": [
                "invest", "investment", "roi", "return", "rental", "rent",
                "income", "profit", "cash flow", "appreciation", "cap rate",
                "yield", "portfolio", "property value", "market value",
                "flip", "flipping", "renovate", "renovation", "fixer",
                "tenant", "tenants", "lease", "passive income"
            ],
            "phrases": [
                "investment property", "rental income", "cash flow",
                "property investment", "real estate investment", "buy to rent",
                "investment opportunity", "rental property", "fix and flip",
                "investment potential", "capital appreciation"
            ],
            "anti_keywords": []
        },
        "young_professional": {
            "keywords": [
                "downtown", "urban", "city", "transit", "subway", "metro",
                "walkable", "walk", "bike", "nightlife", "bars", "restaurants",
                "coffee", "cafe", "gym", "fitness", "amenities", "modern",
                "contemporary", "loft", "studio", "condo", "apartment",
                "coworking", "commute", "work", "professional", "young",
                "trendy", "hip", "vibrant", "social", "entertainment"
            ],
            "phrases": [
                "close to work", "near downtown", "public transit",
                "walking distance", "bike friendly", "night life",
                "young professional", "city life", "urban living",
                "easy commute", "coffee shops", "work from home"
            ],
            "anti_keywords": []
        }
    }

    def __init__(self, llm):
        self.llm = llm
        self.agent = create_agent(
            llm,
            system_prompt="""You are an intelligent query analyzer for real estate searches.

Your task: Analyze the user's query and determine which agent types are most relevant.

Agent Types:
- family: Families with children, school needs, safety, suburban living
- investor: Real estate investors, ROI-focused, rental income, property value
- young_professional: Urban lifestyle, downtown, nightlife, transit, modern amenities

Return ONLY a JSON object with confidence scores (0.0 to 1.0):
{"family": 0.85, "investor": 0.1, "young_professional": 0.3}

Rules:
- If query clearly matches one type: give it 0.8-1.0
- If query is ambiguous or general: give similar scores to multiple types (0.4-0.6)
- If query mentions multiple aspects: score all relevant types accordingly
- Minimum score is 0.0, maximum is 1.0

Examples:
"Good schools for my kids" → {"family": 0.95, "investor": 0.1, "young_professional": 0.05}
"High ROI properties" → {"family": 0.1, "investor": 0.95, "young_professional": 0.1}
"Walkable downtown area" → {"family": 0.2, "investor": 0.3, "young_professional": 0.9}
"Nice place to live" → {"family": 0.5, "investor": 0.4, "young_professional": 0.5}
"3 bedroom house" → {"family": 0.6, "investor": 0.5, "young_professional": 0.3}

Return ONLY the JSON object, nothing else."""
        )

    def calculate_keyword_scores(self, question: str) -> Dict[str, float]:
        """Calculate routing scores based on keyword matching"""
        question_lower = question.lower()
        scores = {}

        for agent_type, indicators in self.ROUTING_INDICATORS.items():
            score = 0.0
            max_score = 0.0

            # Check keywords (1 point each)
            keyword_matches = sum(1 for keyword in indicators["keywords"]
                                 if keyword in question_lower)
            score += keyword_matches * 1.0
            max_score += len(indicators["keywords"])

            # Check phrases (2 points each, more weight)
            phrase_matches = sum(1 for phrase in indicators["phrases"]
                                if phrase in question_lower)
            score += phrase_matches * 2.0
            max_score += len(indicators["phrases"]) * 2.0

            # Normalize to 0-1 range
            if max_score > 0:
                normalized_score = min(score / (max_score * 0.1), 1.0)  # Scale factor
            else:
                normalized_score = 0.0

            scores[agent_type] = normalized_score

        return scores

    def infer_llm(self, question: str) -> Dict[str, float]:
        """Use LLM to analyze query and return confidence scores"""
        try:
            res = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )

            output = res['messages'][-1].content
            print(f"LLM Router Output: {output}")

            # Parse JSON response
            import json
            # Clean up potential markdown formatting
            output = re.sub(r'```json\s*|\s*```', '', output, flags=re.IGNORECASE)
            output = output.strip()

            confidences = json.loads(output)

            # Ensure all agent types are present
            for agent_type in self.AVAILABLE_MODELS:
                if agent_type not in confidences:
                    confidences[agent_type] = 0.0

            return confidences

        except Exception as e:
            print(f"Error in LLM routing: {e}")
            # Return neutral scores if LLM fails
            return {agent: 0.5 for agent in self.AVAILABLE_MODELS}

    def combine_scores(self, keyword_scores: Dict[str, float],
                       llm_scores: Dict[str, float],
                       keyword_weight: float = 0.4,
                       llm_weight: float = 0.6) -> Dict[str, float]:
        """Combine keyword-based and LLM-based scores"""
        combined = {}

        for agent_type in self.AVAILABLE_MODELS:
            kw_score = keyword_scores.get(agent_type, 0.0)
            llm_score = llm_scores.get(agent_type, 0.5)

            # Weighted average
            combined[agent_type] = (kw_score * keyword_weight +
                                   llm_score * llm_weight)

        return combined

    def select_models(self, question: str,
                     confidence_threshold: float = 0.45,
                     ambiguity_threshold: float = 0.25) -> Tuple[Dict[str, float], List[str]]:
        """
        Select which agent(s) should handle the query.

        Args:
            question: User's query
            confidence_threshold: Minimum score to select an agent
            ambiguity_threshold: If all scores are within this range, query all agents

        Returns:
            (confidences_dict, selected_agents_list)
        """
        # Step 1: Calculate keyword-based scores
        keyword_scores = self.calculate_keyword_scores(question)
        print(f"Keyword scores: {keyword_scores}")

        # Step 2: Get LLM-based scores
        llm_scores = self.infer_llm(question)
        print(f"LLM scores: {llm_scores}")

        # Step 3: Combine scores
        final_scores = self.combine_scores(keyword_scores, llm_scores)
        print(f"Final combined scores: {final_scores}")

        # Step 4: Determine ambiguity
        max_score = max(final_scores.values())
        min_score = min(final_scores.values())
        score_range = max_score - min_score

        selected_agents = []

        # Check if query is ambiguous (all scores are similar)
        if score_range < ambiguity_threshold:
            print(f"Ambiguous query detected (score range: {score_range:.3f})")
            print("Routing to ALL agents for comprehensive response")
            selected_agents = self.AVAILABLE_MODELS.copy()
        else:
            # Select agents above threshold
            for agent_type, score in final_scores.items():
                if score >= confidence_threshold:
                    selected_agents.append(agent_type)

            # If no agents selected, use the highest scoring one
            if not selected_agents:
                best_agent = max(final_scores.items(), key=lambda x: x[1])
                selected_agents.append(best_agent[0])
                print(f"No agents above threshold, using best match: {best_agent[0]} ({best_agent[1]:.3f})")

        print(f"Selected agents: {selected_agents}")

        return final_scores, selected_agents

    def is_ambiguous_query(self, scores: Dict[str, float],
                          threshold: float = 0.25) -> bool:
        """Check if the query is ambiguous based on score distribution"""
        max_score = max(scores.values())
        min_score = min(scores.values())
        return (max_score - min_score) < threshold
