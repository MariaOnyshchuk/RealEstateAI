from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, Optional, List
import json
import re
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent import Agent


class Property(BaseModel):
    """Model for property data with validation"""
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    address: str
    price: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    size: Optional[float] = None
    property_type: Optional[str] = None
    roi: Optional[float] = None
    property_url: str = ""

    @field_validator('price', 'size', 'roi', mode='before')
    @classmethod
    def convert_to_float(cls, v):
        """Convert string numbers to float"""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            # Remove currency symbols and commas
            cleaned = re.sub(r'[$,€£¥]', '', v.strip())
            try:
                return float(cleaned)
            except ValueError:
                return None
        return float(v)

    @field_validator('bedrooms', 'bathrooms', mode='before')
    @classmethod
    def convert_to_int(cls, v):
        """Convert string numbers to int"""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            try:
                return int(float(v))
            except ValueError:
                return None
        return int(v)


class AgentResponse(BaseModel):
    """Model for agent response with validation"""
    model_config = ConfigDict(extra="allow")

    summary: str = ""
    top_properties: List[Property] = Field(default_factory=list)


def parse_agent_json(answer: str) -> AgentResponse:
    """
    Parse JSON from agent response with better error handling
    """
    try:
        # Remove markdown code blocks
        cleaned = re.sub(r'```json\s*|\s*```', '', answer, flags=re.IGNORECASE)

        # Try to find JSON object
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)

        if not json_match:
            print("No JSON found in response")
            return AgentResponse(
                summary="Error: No valid JSON response from agent",
                top_properties=[]
            )

        json_str = json_match.group(0)

        # Parse JSON
        data = json.loads(json_str)

        # Validate and return
        return AgentResponse.model_validate(data)

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Attempted to parse: {answer[:200]}...")
        return AgentResponse(
            summary=f"Error parsing response: {str(e)}",
            top_properties=[]
        )
    except Exception as e:
        print(f"Unexpected error: {e}")
        return AgentResponse(
            summary=f"Error: {str(e)}",
            top_properties=[]
        )


class ParameterExtractor:
    """Extract search parameters from natural language queries"""

    def __init__(self, llm):
        self.llm = llm
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a parameter extraction system for real estate searches.
Extract ALL relevant information from the user's query, even if they use non-standard terms.
Return ONLY valid JSON in this format:

{{
    "location": "city/neighborhood/area or null",
    "price_min": numeric value or null,
    "price_max": numeric value or null,
    "bedrooms_min": number or null,
    "bedrooms_max": number or null,
    "bathrooms_min": number or null,
    "bathrooms_max": number or null,
    "size_min": square footage or null,
    "size_max": square footage or null,
    "property_type": "house/apartment/condo/townhouse/studio or null",
    "amenities": ["pool", "gym", "parking", "etc"],
    "preferences": ["quiet", "walkable", "modern", "etc"],
    "lifestyle": "family/investor/young_professional/luxury/budget or null",
    "keywords": ["any", "other", "relevant", "terms"]
}}

IMPORTANT: Be flexible and interpret abstract requests:
- "cozy" → smaller size, preferences: ["cozy", "intimate"]
- "spacious" → larger size
- "affordable" → lower price range
- "luxury" → higher price, amenities: ["high-end"]
- "near downtown" → location context, preferences: ["central"]
- "good schools" → lifestyle: "family", preferences: ["schools"]
- "investment" → lifestyle: "investor"
- "walkable" → preferences: ["walkable", "pedestrian-friendly"]
- "modern" → preferences: ["modern", "contemporary"]
- "garden" → amenities: ["garden", "outdoor space"]

Extract ranges when mentioned (e.g., "2-3 bedrooms" → bedrooms_min: 2, bedrooms_max: 3)
Return null for unmentioned fields. All numbers should be numeric, not strings."""),
            ("user", "{query}")
        ])

    def extract(self, query: str) -> Dict[str, Any]:
        """Extract search parameters from natural language query"""
        try:
            chain = self.extraction_prompt | self.llm
            response = chain.invoke({"query": query})

            # Get content from response
            content = getattr(response, "content", str(response)).strip()

            # Clean markdown formatting
            content = re.sub(r'```json\s*|\s*```', '', content, flags=re.IGNORECASE)

            # Parse JSON
            result = json.loads(content)

            # Normalize and validate extracted data
            result = self._normalize_parameters(result, query)

            print("Extracted parameters:", json.dumps(result, indent=2))
            return result

        except json.JSONDecodeError as e:
            print(f"Failed to parse extraction result: {e}")
            # Return a basic fallback with the original query
            return {
                "keywords": [query],
                "raw_query": query
            }
        except Exception as e:
            print(f"Extraction error: {e}")
            return {
                "keywords": [query],
                "raw_query": query
            }

    def _normalize_parameters(self, params: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Normalize and add context to extracted parameters"""
        # Keep original query for reference
        params["raw_query"] = original_query

        # Ensure lists exist
        if "amenities" not in params:
            params["amenities"] = []
        if "preferences" not in params:
            params["preferences"] = []
        if "keywords" not in params:
            params["keywords"] = []

        # Add common interpretations if not already captured
        query_lower = original_query.lower()

        # Detect lifestyle hints
        if not params.get("lifestyle"):
            if any(word in query_lower for word in ["kid", "child", "family", "school"]):
                params["lifestyle"] = "family"
            elif any(word in query_lower for word in ["invest", "roi", "rental"]):
                params["lifestyle"] = "investor"
            elif any(word in query_lower for word in ["downtown", "nightlife", "transit", "walkable"]):
                params["lifestyle"] = "young_professional"

        # Add price context for vague terms
        if any(term in query_lower for term in ["cheap", "affordable", "budget"]) and not params.get("price_max"):
            params["preferences"].append("budget-friendly")

        if any(term in query_lower for term in ["luxury", "high-end", "premium"]) and not params.get("price_min"):
            params["preferences"].append("luxury")

        return params


class BaseAgent:
    """Base class for specialized real estate agents"""

    def __init__(self, llm, agent_type: str):
        self.llm = llm
        self.agent_type = agent_type
        self.sql_agent = Agent(llm, agent_type)
        self.system_prompt = "You are a helpful real estate assistant."

    def set_prompt(self, prompt_text: str):
        """Set the system prompt for the agent"""
        self.system_prompt = prompt_text

    def infer(self, question: str, search_params: Optional[Dict] = None) -> AgentResponse:
        """
        Process a question and return structured results
        Handles both structured parameters and free-form queries
        """
        # Build enriched question with context
        context_parts = []

        if search_params:
            # Format parameters in a clear, readable way for the LLM
            context_parts.append("Search Parameters:")

            # Location
            if search_params.get("location"):
                context_parts.append(f"- Location: {search_params['location']}")

            # Price range
            price_parts = []
            if search_params.get("price_min"):
                price_parts.append(f"minimum ${search_params['price_min']:,}")
            if search_params.get("price_max"):
                price_parts.append(f"maximum ${search_params['price_max']:,}")
            if price_parts:
                context_parts.append(f"- Price: {' to '.join(price_parts)}")

            # Bedrooms/Bathrooms
            if search_params.get("bedrooms_min") or search_params.get("bedrooms_max"):
                bed_range = self._format_range(
                    search_params.get("bedrooms_min"),
                    search_params.get("bedrooms_max"),
                    "bedroom"
                )
                context_parts.append(f"- Bedrooms: {bed_range}")

            if search_params.get("bathrooms_min") or search_params.get("bathrooms_max"):
                bath_range = self._format_range(
                    search_params.get("bathrooms_min"),
                    search_params.get("bathrooms_max"),
                    "bathroom"
                )
                context_parts.append(f"- Bathrooms: {bath_range}")

            # Size
            if search_params.get("size_min") or search_params.get("size_max"):
                size_range = self._format_range(
                    search_params.get("size_min"),
                    search_params.get("size_max"),
                    "sqft"
                )
                context_parts.append(f"- Size: {size_range}")

            # Property type
            if search_params.get("property_type"):
                context_parts.append(f"- Type: {search_params['property_type']}")

            # Amenities
            if search_params.get("amenities"):
                context_parts.append(f"- Required amenities: {', '.join(search_params['amenities'])}")

            # Preferences
            if search_params.get("preferences"):
                context_parts.append(f"- Preferences: {', '.join(search_params['preferences'])}")

            # Lifestyle
            if search_params.get("lifestyle"):
                context_parts.append(f"- Lifestyle match: {search_params['lifestyle']}")

            # Keywords and raw query
            if search_params.get("keywords"):
                context_parts.append(f"- Additional keywords: {', '.join(search_params['keywords'])}")

            if search_params.get("raw_query"):
                context_parts.append(f"- Original request: \"{search_params['raw_query']}\"")

        context = "\n".join(context_parts) if context_parts else "No specific parameters provided"

        enriched_question = f"""{self.system_prompt}

{context}

User Question: {question}

IMPORTANT INSTRUCTIONS:
1. Query the database flexibly - use approximate matches for text fields
2. If exact matches aren't found, find the closest alternatives
3. Consider all mentioned preferences, amenities, and lifestyle factors
4. Explain in the summary if you made adjustments to the search criteria
5. Return your response as valid JSON in this exact format:

{{
    "summary": "Brief explanation of results and any adjustments made to search criteria",
    "top_properties": [
        {{
            "address": "full address",
            "price": 250000,
            "bedrooms": 3,
            "bathrooms": 2,
            "size": 1500,
            "property_type": "house",
            "roi": 5.2,
            "property_url": "url if available"
        }}
    ]
}}

Return 5-10 properties that best match the criteria. Include properties even if they don't match perfectly."""

        try:
            # Get response from SQL agent
            answer = self.sql_agent.infer(enriched_question)

            # Parse the response
            parsed_response = parse_agent_json(answer)

            return parsed_response

        except Exception as e:
            print(f"Error in agent inference: {e}")
            return AgentResponse(
                summary=f"Error processing request: {str(e)}",
                top_properties=[]
            )

    def _format_range(self, min_val, max_val, unit: str) -> str:
        """Format a range for display"""
        if min_val==max_val:
            return f"{min_val} {unit}{'s' if min_val%10!=1 else ''}"
        elif min_val and max_val:
            return f"{min_val}-{max_val} {unit}s"
        elif min_val:
            return f"{min_val}+ {unit}s"
        elif max_val:
            return f"up to {max_val} {unit}s"
        return "any"


class FamilyAgent(BaseAgent):
    """Real estate agent specialized for family needs"""

    def __init__(self, llm):
        super().__init__(llm, 'family')
        self.set_prompt("""You are a Real Estate Agent specializing in Family needs.

Focus on:
- School quality and proximity
- Safety and crime rates
- Parks and outdoor spaces
- Quiet, family-friendly neighborhoods
- Backyard space and home size
- Community events and activities

Tone: Warm, reassuring, and family-focused.
Highlight properties that offer great environments for raising children.""")


class InvestorAgent(BaseAgent):
    """Real estate agent specialized for investors"""

    def __init__(self, llm):
        super().__init__(llm, 'investor')
        self.set_prompt("""You are a Real Estate Investment Analyst.

Focus on:
- ROI and cap rates
- Rental income potential
- Property appreciation trends
- Up-and-coming neighborhoods
- Market analysis and comparables
- Cash flow projections

Tone: Analytical, data-driven, and concise.
Provide investment metrics and financial insights.""")


class YoungProfessionalAgent(BaseAgent):
    """Real estate agent specialized for young professionals"""

    def __init__(self, llm):
        super().__init__(llm, 'young_professional')
        self.set_prompt("""You are a Real Estate Agent for Young Professionals.

Focus on:
- Nightlife and entertainment
- Coworking spaces and cafes
- Modern amenities (gym, pool, etc.)
- Public transit access
- Walkability and bike-friendliness
- Trendy neighborhoods

Tone: Energetic, modern, and lifestyle-focused.
Emphasize urban convenience and social opportunities.""")


# Example usage function
def display_results_as_table(response: AgentResponse) -> str:
    """Convert agent response to a formatted table string"""
    if not response.top_properties:
        return f"Summary: {response.summary}\n\nNo properties found."

    # Create table header
    table = f"Summary: {response.summary}\n\n"
    table += f"{'Address':<40} {'Price':<15} {'Beds':<6} {'Baths':<6} {'Size':<10} {'Type':<15} {'ROI':<8}\n"
    table += "-" * 115 + "\n"

    # Add property rows
    for prop in response.top_properties:
        price_str = f"${prop.price:,.0f}" if prop.price else "N/A"
        size_str = f"{prop.size:.0f} sqft" if prop.size else "N/A"
        roi_str = f"{prop.roi:.1f}%" if prop.roi else "N/A"
        beds = str(prop.bedrooms) if prop.bedrooms else "N/A"
        baths = str(prop.bathrooms) if prop.bathrooms else "N/A"
        prop_type = prop.property_type or "N/A"

        table += f"{prop.address:<40} {price_str:<15} {beds:<6} {baths:<6} {size_str:<10} {prop_type:<15} {roi_str:<8}\n"

    return table


def process_user_query(query: str, llm, agent_type: str = "auto") -> AgentResponse:
    """
    Complete pipeline to process any user query format
    Automatically selects appropriate agent based on query content
    """
    # Extract parameters from natural language
    extractor = ParameterExtractor(llm)
    params = extractor.extract(query)

    # Auto-select agent based on lifestyle detection or use specified type
    if agent_type == "auto":
        lifestyle = params.get("lifestyle", "").lower()
        if "family" in lifestyle:
            agent = FamilyAgent(llm)
        elif "investor" in lifestyle or "investment" in lifestyle:
            agent = InvestorAgent(llm)
        elif "young_professional" in lifestyle or "professional" in lifestyle:
            agent = YoungProfessionalAgent(llm)
        else:
            # Default to young professional for general queries
            agent = YoungProfessionalAgent(llm)
    else:
        # Use specified agent type
        if agent_type == "family":
            agent = FamilyAgent(llm)
        elif agent_type == "investor":
            agent = InvestorAgent(llm)
        else:
            agent = YoungProfessionalAgent(llm)

    # Get results
    response = agent.infer(query, search_params=params)

    return response


# Example usage with various input formats:
"""
Example 1 - Precise query:
query = "Find me a 3 bedroom house in Seattle under $500,000 with a backyard"
response = process_user_query(query, llm)

Example 2 - Abstract query:
query = "I want something cozy near downtown where I can walk to coffee shops"
response = process_user_query(query, llm)

Example 3 - Lifestyle-based query:
query = "Looking for a good investment property that will generate rental income"
response = process_user_query(query, llm, agent_type="investor")

Example 4 - Vague query:
query = "Show me nice places to live"
response = process_user_query(query, llm)

Example 5 - Query with non-standard terms:
query = "Need a pad with sick views, close to the action, modern vibes"
response = process_user_query(query, llm)

Example 6 - Family-oriented abstract query:
query = "We have two kids and want somewhere safe with good schools"
response = process_user_query(query, llm)

# Display results
print(display_results_as_table(response))

# Or use structured data
for prop in response.top_properties:
    print(f"{prop.address}: ${prop.price:,.0f}")
"""