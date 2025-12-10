from langchain_community.utilities import SQLDatabase
from typing import Literal
import json
import re
from langgraph.prebuilt import ToolNode
from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field
from typing import List, Optional
from property_retrieval import QueryResult
from agents.prompts import CHECK_QUERY, GENERATE_QUERY
from agents.prompts import FORMAT_PROMPT, FAMIILY_AGENT, INVESTOR_AGENT, YOUNG_PROFESSIONAL

import os, dotenv

AGENT_EXTRA_FIELDS = {
    "family": ["school", "park", "pharmacy", "supermarket", "crime_rate"],
    "investor": ["avg_monthly_rent", "tax"],
    "young_professional": ["cafe", "restaurant", "gym", "library"]
}


AGENT_PROMPTS = {
        "family": FAMIILY_AGENT,
        "investor": INVESTOR_AGENT,
        "young_professional": YOUNG_PROFESSIONAL
    }


class Agent:
    def __init__(self, llm, agent_type):
        from langchain_community.agent_toolkits import SQLDatabaseToolkit

        self.llm = llm
        self.agent_type = agent_type

        # --- Database setup ---
        dotenv.load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        self.db = SQLDatabase.from_uri(db_url)

        # self.db = SQLDatabase.from_uri("sqlite:///data.db") // for local use
        self.dialect = self.db.dialect
        self.top_k = 3

        # --- Toolkit and tools ---
        toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
        tools = toolkit.get_tools()
        self.tools = tools

        self.get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
        self.run_query_tool = next(t for t in tools if t.name == "sql_db_query")

        self.get_schema_node = ToolNode([self.get_schema_tool], name="get_schema")
        self.run_query_node = ToolNode([self.run_query_tool], name="run_query")
        self.table_name = 'DB'

        self.generate_query_system_prompt = GENERATE_QUERY.format(
            agent_type=self.agent_type,
            dialect=self.dialect,
            table_name=self.table_name,
            top_k=self.top_k
        )

        self.check_query_system_prompt = CHECK_QUERY.format(
            table_name=self.table_name,
            top_k = self.top_k
        )

        def list_tables(state: MessagesState):
            """
            List all usable tables in the database.
            Returns a message containing available tables.
            """
            list_tables_tool = next(t for t in self.tools if t.name == "sql_db_list_tables")

            tool_call = {"name": list_tables_tool.name, "args": {}, "id": "list_tables_call", "type": "tool_call"}
            tool_response = list_tables_tool.invoke(tool_call)

            response_message = AIMessage(content=f"Available tables: {tool_response.content}")

            return {"messages": [response_message]}


        def call_get_schema(state: MessagesState):
            # FIX: Explicitly pass the table name to get_schema_tool
            tool_call = {
                "name": self.get_schema_tool.name,
                "args": {"table_names": self.table_name},  # Add this!
                "id": "get_schema_call",
                "type": "tool_call"
            }

            # Create an AIMessage with the tool call
            ai_message = AIMessage(content="", tool_calls=[tool_call])

            return {"messages": [ai_message]}
        def generate_query(state: MessagesState):
            """
            Generate SQL query based on user request and schema.
            """
            print(f"\n{'='*50}\nGenerating SQL Query\n{'='*50}\n")

            system_message = {
                "role": "system",
                "content": self.generate_query_system_prompt,
            }

            # Bind the run_query_tool so LLM knows it should call it
            llm_with_tools = self.llm.bind_tools([self.run_query_tool])

            # Invoke with system message + conversation history
            response = llm_with_tools.invoke([system_message] + state["messages"])

            print(f"LLM Response Type: {type(response)}")
            print(f"Has tool_calls: {hasattr(response, 'tool_calls')}")

            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"✅ Generated {len(response.tool_calls)} tool call(s)")
                print(f"Tool call: {response.tool_calls[0]}")
            else:
                print("❌ No tool calls generated!")
                print(f"Response content: {getattr(response, 'content', 'No content')[:200]}")

            return {"messages": [response]}

        def validate_query(state: MessagesState):
            """
            Validate the generated SQL query for safety and correctness.
            Block dangerous operations and check for required fields.
            """
            last_msg = state["messages"][-1]
            if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
                # No query generated - this is an error state
                error_msg = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": self.run_query_tool.name,
                        "args": {"query": "SELECT 'Error: No query generated' as error"},
                        "id": "error_query",
                        "type": "tool_call"
                    }]
                )
                return {"messages": [error_msg]}

            query = last_msg.tool_calls[0]["args"]["query"]
            query_lower = query.lower().strip()

            print(f"\n{'='*50}\nValidating SQL Query:\n{query}\n{'='*50}\n")

            # 1. Block dangerous operations
            dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'truncate', 'alter', 'create']
            for keyword in dangerous_keywords:
                if f' {keyword} ' in f' {query_lower} ' or query_lower.startswith(keyword):
                    print(f"⚠️  BLOCKED: Query contains dangerous keyword '{keyword}'")
                    error_msg = AIMessage(
                        content="",
                        tool_calls=[{
                            "name": self.run_query_tool.name,
                            "args": {"query": f"SELECT 'Error: Dangerous operation blocked ({keyword})' as error"},
                            "id": "blocked_query",
                            "type": "tool_call"
                        }]
                    )
                    return {"messages": [error_msg]}

            # 2. Verify it's a SELECT query
            if not query_lower.strip().startswith('select'):
                print(f"!!  WARNING: Query doesn't start with SELECT")

            # 3. Check table name
            table_check = f'from {self.table_name.lower()}'
            if table_check not in query_lower and f'from `{self.table_name.lower()}`' not in query_lower:
                print(f"!!  WARNING: Query might not use correct table name '{self.table_name}'")

            # 4. Verify required fields are present
            required_fields = ['property_url', 'full_street_line', 'list_price', 'beds', 'full_baths', 'sqft', 'style', 'roi']
            required_fields.extend(AGENT_EXTRA_FIELDS.get(self.agent_type, []))

            missing_fields = []
            for field in required_fields:
                if field.lower() not in query_lower:
                    missing_fields.append(field)

            if missing_fields:
                print(f"‼ WARNING: Missing required fields: {', '.join(missing_fields)}")

            # Query passed validation - return it as-is
            print("✅ Query validation passed")
            return {"messages": [last_msg]}

        def format_final_results(state: MessagesState):
            """
            Format SQL results into the final QueryResult JSON structure.
            Combines formatting and validation in one step.
            """
            messages = state["messages"]

            # Find the SQL results from ToolMessage
            sql_results = None
            for msg in reversed(messages):
                if hasattr(msg, '__class__') and msg.__class__.__name__ == 'ToolMessage':
                    sql_results = msg.content
                    break

            # Handle empty or error results
            if not sql_results or sql_results.strip() in ["[]", "", "None", "null"]:
                print("ℹ️  No SQL results found")
                result_json = QueryResult(
                    summary="No properties found matching your criteria.",
                    top_properties=[]
                )
                return {"messages": [AIMessage(content=result_json.model_dump_json())]}

            # Check for error messages from validation
            if "error" in sql_results.lower() and "blocked" in sql_results.lower():
                print("⚠️  Query was blocked")
                result_json = QueryResult(
                    summary="Query could not be executed due to safety restrictions.",
                    top_properties=[]
                )
                return {"messages": [AIMessage(content=result_json.model_dump_json())]}

            print(f"\n{'='*50}\nFormatting SQL Results:\n{sql_results[:500]}...\n{'='*50}\n")

            # Get agent-specific instructions and fields
            agent_summary_style = AGENT_PROMPTS.get(self.agent_type, "You are a real estate agent.")

            user_query = ""
            for msg in state["messages"]:
                if hasattr(msg, 'role') and msg.role == 'user':
                    user_query = msg.content
                    break
                elif isinstance(msg, dict) and msg.get('role') == 'user':
                    user_query = msg.get('content', '')
                    break

            # Build the field schema for the LLM
            field_schema = []
            field_schema.append('"property_url": "string (required)"')
            field_schema.append('"full_street_line": "string (required)"')
            field_schema.append('"list_price": number (required)')
            field_schema.append('"beds": number (required)')
            field_schema.append('"full_baths": number (required)')
            field_schema.append('"sqft": number (required)')
            field_schema.append('"style": "string (required)"')
            field_schema.append('"roi": number')
            field_schema.append('"school": number')
            field_schema.append('"park": number')
            field_schema.append('"pharmacy": number')
            field_schema.append('"supermarket": number')
            field_schema.append('"crime_rate": number')
            field_schema.append('"gym": number')
            field_schema.append('"library": number')
            field_schema.append('"restaurant": number')
            field_schema.append('"tax": number')
            field_schema.append('"avg_monthly_rent": number')
            field_schema.append('"night_club": number')
            field_schema.append('"town_square": number')
            field_schema.append('"stories": number')
            field_schema_str = ",\n    ".join(field_schema)

            format_prompt = f"""You are a {self.agent_type} real estate agent formatting search results.

AGENT PERSONALITY & SUMMARY STYLE:
{agent_summary_style}

USER ASKED FOR:
"{user_query}"


SQL Results:
{sql_results}
Here is the schema of the results. You can map them via index:
{field_schema_str}

YOUR TASK:
1. Write a summary that matches your agent personality and highlights what matters most to your target audience
2. Format all properties into the JSON structure below

CRITICAL - SUMMARY REQUIREMENTS:
- Use the tone and style described in "AGENT PERSONALITY & SUMMARY STYLE" above
- Highlight the key features that matter to a {self.agent_type}
- Be specific about what was found (number of properties, price ranges, key features)
- Keep it 1-3 sentences maximum


{{
"summary": "Your personalized summary here following the agent style guidelines",
"top_properties": [
    {{
    {field_schema_str}
    }}
]
}}

CRITICAL RULES:
1. Return ONLY valid JSON, no explanation or markdown
2. All numeric fields (list_price, beds, sqft, roi, etc.) must be numbers or 0 if null, not strings
3. If a field is missing or empty in the SQL results, set it to null
4. Property_url must never be null or empty string
5. Include ALL properties from the SQL results (up to {self.top_k})
6. Do not add fields that weren't requested
7. No ```json``` code blocks - just pure JSON

Please preserve all of the variables in the JSON as in the example:
{{
"summary": "These homes offer a welcoming environment for families looking for comfort and stability. Each property features spacious layouts ideal for everyday living, with strong nearby school options and easy access to parks and outdoor spaces. Located in quiet, family-friendly neighborhoods, they provide a safe atmosphere, room to grow, and the kind of community where children can thrive.",
"top_properties": [
    {{
    "property_url": "https://example.com/property/123",
    "full_street_line": "123 Main St",
    "list_price": 450000,
    "beds": 3,
    "full_baths": 2,
    "sqft": 1800,
    "style": "SINGLE_FAMILY",
    "roi": 5.2
    "school": 8,
    "park": 9,
    "pharmacy": 5,
    "supermarket": 7,
    "crime_rate": 375.8,
    "gym": 6,
    "library": 4,
    "restaurant": 8,
    "tax": 1691.0,
    "avg_monthly_rent": 2916.434316353887,
    "night_club": 5,
    "town_square": 2,
    "stories": 1
    }}
]
}}
"""

            # Get LLM response
            response = self.llm.invoke([{"role": "user", "content": format_prompt}])
            content = getattr(response, 'content', str(response))

            # Clean up the response
            content_clean = content.strip()
            content_clean = re.sub(r'```json\s*|\s*```', '', content_clean, flags=re.IGNORECASE).strip()

            print(f"\n{'='*50}\nLLM Formatted Response:\n{content_clean}...\n{'='*50}\n")

            # Parse and validate JSON
            try:
                parsed_dict = json.loads(content_clean)

                # Define all numeric fields
                numeric_fields = {"list_price", "beds", "full_baths", "half_baths", "sqft", "roi"}
                numeric_fields.update(AGENT_EXTRA_FIELDS.get(self.agent_type, []))

                # Clean and validate each property
                for prop in parsed_dict.get("top_properties", []):
                    # Convert numeric fields
                    for key in numeric_fields:
                        if key in prop:
                            val = prop[key]
                            if val is None or val == "" or val == "null":
                                prop[key] = None
                            else:
                                try:
                                    prop[key] = float(val) if isinstance(val, str) else val
                                except (ValueError, TypeError):
                                    prop[key] = None

                    # Ensure property_url exists and is not empty
                    if "property_url" not in prop or not prop["property_url"]:
                        prop["property_url"] = ""

                # Create validated result
                result_json = QueryResult(**parsed_dict)
                print(f"✅ Successfully formatted {len(result_json.top_properties)} properties")

                return {"messages": [AIMessage(content=result_json.model_dump_json())]}

            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"Failed content: {content_clean[:200]}")

                result_json = QueryResult(
                    summary="Error formatting results. Please try rephrasing your query.",
                    top_properties=[]
                )
                return {"messages": [AIMessage(content=result_json.model_dump_json())]}

            except Exception as e:
                print(f"❌ Unexpected error: {e}")

                result_json = QueryResult(
                    summary="An error occurred while processing results.",
                    top_properties=[]
                )
                return {"messages": [AIMessage(content=result_json.model_dump_json())]}

        builder = StateGraph(MessagesState)

        builder.add_node("list_tables", list_tables)
        builder.add_node("call_get_schema", call_get_schema)
        builder.add_node("get_schema", self.get_schema_node)
        builder.add_node("generate_query", generate_query)
        # builder.add_node("check_query", check_query)
        builder.add_node("validate_query", validate_query)
        builder.add_node("run_query", self.run_query_node)
        # builder.add_node("format_query_results", format_query_results)  # NEW NODE
        # builder.add_node("format_results", format_results)
        builder.add_node("format_final_results", format_final_results)

        builder.add_edge(START, "list_tables")
        builder.add_edge("list_tables", "call_get_schema")
        builder.add_edge("call_get_schema", "get_schema")
        builder.add_edge("get_schema", "generate_query")
        # builder.add_edge("generate_query", "check_query")
        # builder.add_edge("check_query", "run_query")
        builder.add_edge("generate_query", "validate_query")
        builder.add_edge("validate_query", "run_query")
        # builder.add_edge("run_query", "format_query_results")  # NEW EDGE
        # builder.add_edge("format_query_results", "format_results")  # NEW EDGE
        # builder.add_edge("format_results", END)
        builder.add_edge("run_query", "format_final_results")
        builder.add_edge("format_final_results", END)

        self.agent = builder.compile()


    def infer(self, prompt: str) -> dict:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        content = result["messages"][-1].content.strip()
        content = re.sub(r"```json|```", "", content).strip()

        return json.loads(content)

