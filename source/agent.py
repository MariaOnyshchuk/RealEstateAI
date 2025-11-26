from langchain_community.utilities import SQLDatabase

from typing import Literal
import json
import re
from langgraph.prebuilt import ToolNode
from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

class Agent:

    def __init__(self, llm, agent_type):

        db = SQLDatabase.from_uri("sqlite:///data.db")

        print(f"Dialect: {db.dialect}")
        print(f"Available tables: {db.get_usable_table_names()}")
        # print(f'Sample output: {db.run("SELECT * FROM DB LIMIT 5;")}')

        from langchain_community.agent_toolkits import SQLDatabaseToolkit

        self.llm = llm
        self.db = db
        self.agent_type = agent_type
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)

        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        tools = toolkit.get_tools()

        for tool in tools:
            print(f"{tool.name}: {tool.description}\n")

        self.tools = tools

        self.get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
        self.get_schema_node = ToolNode([self.get_schema_tool], name="get_schema")

        self.run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
        self.run_query_node = ToolNode([self.run_query_tool], name="run_query")
        dialect=db.dialect,
        top_k=10,
        self.generate_query_system_prompt = f"""You are an agent designed to interact with a SQL database for real estate listings.
You specialize in finding properties suitable for: {agent_type}

Given an input question, create a syntactically correct {dialect} query to run, then look at the results and return a JSON response.

DATABASE SCHEMA REMINDER:
- The main table is likely named 'listings', 'properties', or 'DB'
- Common columns: address, price, bedrooms, bathrooms, size, property_type, property_url, roi
- ALWAYS verify column names from the schema before writing queries

IMPORTANT RULES:
1. Always limit results to at most {top_k} properties
2. ALWAYS include the `property_url` column in SELECT statements
3. Order results by relevance (e.g., ROI for investors, price for budget queries)
4. Never make DML statements (INSERT, UPDATE, DELETE, DROP)
5. Only query relevant columns, not all columns
6. Use LIKE '%keyword%' for flexible text matching (case-insensitive)
7. Always use WHERE clauses to filter - avoid full table scans
8. Test for NULL values when filtering (e.g., WHERE column IS NOT NULL)

COMMON QUERY PATTERNS:
- Location search: WHERE LOWER(address) LIKE LOWER('%location%')
- Price range: WHERE price BETWEEN min AND max (or price <= max)
- Bedrooms: WHERE bedrooms >= min AND bedrooms <= max
- Property type: WHERE LOWER(property_type) LIKE LOWER('%type%')
- Combine filters with AND

ERROR PREVENTION:
- Always check if columns exist in schema before using them
- Use table name prefix if multiple tables (e.g., DB.address)
- Quote column names with spaces or special characters
- Avoid division by zero (check for NULL/zero before dividing)
- Use COALESCE for NULL handling: COALESCE(roi, 0)

RESPONSE FORMAT - Return ONLY valid JSON:
{{
    "summary": "Brief explanation of the search results and criteria used",
    "top_properties": [
        {{
            "address": "full address",
            "price": 250000.00,
            "bedrooms": 3,
            "bathrooms": 2,
            "size": 1500.0,
            "property_type": "house",
            "roi": 5.5,
            "property_url": "https://example.com/property"
        }}
    ]
}}

SEARCH FLEXIBILITY:
- If searching by location, use LIKE '%location%' for flexible matching
- For price ranges, use BETWEEN or <= operators
- For bedrooms/bathrooms, allow some flexibility (e.g., 2-4 bedrooms instead of exactly 3)
- For "cozy", interpret as smaller size (< 1500 sqft)
- For "downtown", search in location/address fields with LIKE
- If no exact matches, widen the search criteria

AGENT-SPECIFIC PRIORITIES:
- family: Focus on bedrooms >= 3, look for keywords in address, prioritize safety
- investor: Order by ROI DESC or price/size ratio, focus on rental potential
- young_professional: Look for downtown/urban in address, prefer apartments/condos

After getting SQL results, format them into the JSON structure above. DO NOT TRUNCATE THE JSON. If no results, return empty top_properties array with explanation in summary."""

        self.check_query_system_prompt = f"""You are a SQL expert reviewing {dialect} queries for correctness.

Check for these common mistakes:
- Column names that don't exist in the schema
- Missing table names or wrong table names
- Using NOT IN with NULL values
- Using UNION when UNION ALL is better
- Data type mismatches in comparisons
- Missing quotes around string literals
- Incorrect LIKE syntax (should be LIKE '%text%')
- Missing WHERE clause when filtering is needed
- Dividing by zero or NULL values
- Using aggregate functions without GROUP BY

CRITICAL CHECKS:
1. Verify ALL column names exist in the schema
2. Verify table name is correct (check schema)
3. Ensure all string comparisons use proper quotes
4. Check for NULL handling in calculations
5. Ensure LIKE patterns have % wildcards
6. Ensure that text is not truncated and the query will not cause any problem.

If you find mistakes, rewrite the query correctly.
If the query is correct, reproduce it exactly.

IMPORTANT: After checking, you MUST call the sql_db_query tool to execute the query."""

        def list_tables(state: MessagesState):
            tool_call = {
                "name": "sql_db_list_tables",
                "args": {},
                "id": "abc123",
                "type": "tool_call",
            }
            tool_call_message = AIMessage(content="", tool_calls=[tool_call])

            list_tables_tool = next(tool for tool in self.tools if tool.name == "sql_db_list_tables")
            tool_message = list_tables_tool.invoke(tool_call)
            response = AIMessage(f"Available tables: {tool_message.content}")

            return {"messages": [tool_call_message, tool_message, response]}

        def call_get_schema(state: MessagesState):
            llm_with_tools = self.llm.bind_tools([self.get_schema_tool], tool_choice="any")
            response = llm_with_tools.invoke(state["messages"])

            return {"messages": [response]}

        def generate_query(state: MessagesState):
            system_message = {
                "role": "system",
                "content": self.generate_query_system_prompt,
            }

            llm_with_tools = self.llm.bind_tools([self.run_query_tool])
            response = llm_with_tools.invoke([system_message] + state["messages"])

            return {"messages": [response]}

        def check_query(state: MessagesState):
            system_message = {
                "role": "system",
                "content": self.check_query_system_prompt,
            }

            tool_call = state["messages"][-1].tool_calls[0]
            query = tool_call["args"]["query"]
            query_lower = query.lower()

            dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'truncate', 'alter']
            if any(keyword in query_lower for keyword in dangerous_keywords):
                print(f"BLOCKED: Dangerous query detected: {query}")
                error_response = AIMessage(
                    content="Query contains dangerous operations and was blocked.",
                    tool_calls=[]
                )
                return {"messages": [error_response]}

            if 'select' not in query_lower:
                print(f"WARNING: Query doesn't contain SELECT: {query}")

            if not any(table_word in query_lower for table_word in ['from db', 'from listings', 'from properties']):
                print(f"WARNING: Query might have wrong table name: {query}")

            user_message = {"role": "user", "content": query}

            llm_with_tools = self.llm.bind_tools([self.run_query_tool], tool_choice="any")
            response = llm_with_tools.invoke([system_message, user_message])
            response.id = state["messages"][-1].id

            return {"messages": [response]}

        def format_results(state: MessagesState):
            messages = state["messages"]

            query_result = None
            query_error = None

            for msg in reversed(messages):
                if hasattr(msg, 'content'):
                    content_str = str(msg.content)
                    if any(err in content_str.lower() for err in ['error', 'exception', 'failed', 'invalid']):
                        query_error = content_str
                        break
                    if msg.content and len(content_str) > 10 and 'SELECT' not in content_str.upper():
                        query_result = content_str
                        break

            if query_error or not query_result:
                error_json = {
                    "summary": "We encountered an issue searching the database. This might be due to: (1) No properties matching your exact criteria, or (2) A query error. Try broadening your search parameters.",
                    "top_properties": []
                }
                import json
                response = AIMessage(content=json.dumps(error_json))
                return {"messages": [response]}

            if query_result in ['[]', '()', '', 'None'] or 'no rows' in query_result.lower():
                no_results_json = {
                    "summary": "No properties found matching your exact criteria. Try: (1) Expanding your location search, (2) Adjusting your price range, (3) Being more flexible with bedrooms/bathrooms requirements.",
                    "top_properties": []
                }

                response = AIMessage(content=json.dumps(no_results_json))
                return {"messages": [response]}

            system_message = {
                "role": "system",
                "content": """You are formatting SQL query results into a JSON response.

The user asked a real estate question, and we have SQL results.

Format the results into this EXACT JSON structure:
{{
    "summary": "Explain what was found, search criteria used, and why these properties were selected",
    "top_properties": [
        {{
            "address": "full address as string",
            "price": 250000.00,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "size": 1500.0,
            "property_type": "house",
            "roi": 5.5,
            "property_url": "https://example.com/property"
        }}
    ]
}}

CRITICAL RULES:
- Return ONLY valid JSON, no markdown, no backticks, no extra text
- If query returned no results, return empty top_properties array
- Convert ALL numeric fields to proper types (numbers, not strings)
- Handle NULL/None values by omitting the field or using null
- Include ALL properties from SQL results (up to 10)
- Price should be a float/int, not a string like "$250,000"
- Bedrooms/bathrooms should be integers or floats
- If a field is missing in SQL results, set it to null or omit it
- property_url is REQUIRED - if missing, use empty string

Example transformations:
- "$250,000" → 250000.00
- "3 bedrooms" → 3
- "N/A" → null
- "" → null"""
            }

            format_prompt = f"""SQL Results:
{query_result}
Convert these SQL results into the JSON format specified. Be careful with data types and NULL values."""
            try:
                response = self.llm.invoke([system_message, {"role": "user", "content": format_prompt}])
                # Validate the JSON
                content = response.content
                content = re.sub(r'```json\s*|\s*```', '', content, flags=re.IGNORECASE)

                content = content.strip()
                import json
                try:
                    parsed = json.loads(content)
                    if "summary" not in parsed:
                        parsed["summary"] = "Properties found matching your criteria."
                    if "top_properties" not in parsed:
                        parsed["top_properties"] = []

                    for prop in parsed["top_properties"]:
                        if "address" not in prop:
                            prop["address"] = "Address not available"
                        if "property_url" not in prop:
                            prop["property_url"] = ""

                    response.content = json.dumps(parsed)

                except json.JSONDecodeError as e:
                    print(f"JSON parsing error in format_results: {e}")
                    print(f"Content was: {content}")
                    # Return error JSON
                    error_json = {
                        "summary": "Error formatting results. Please try a different search.",
                        "top_properties": []
                    }
                    response.content = json.dumps(error_json)

            except Exception as e:
                print(f"Error formatting results: {e}")
                error_json = {
                    "summary": f"Error processing query results: {str(e)}",
                    "top_properties": []
                }
                response = AIMessage(content=json.dumps(error_json))

            return {"messages": [response]}


        def should_continue_after_query(state: MessagesState) -> Literal["check_query", "format_results"]:
            """Decide next step after generating query"""
            messages = state["messages"]
            last_message = messages[-1]

            if not last_message.tool_calls:
                return "format_results"

            query_run_count = sum(1 for msg in messages if hasattr(msg, 'name') and msg.name == "sql_db_query")

            if query_run_count > 0:
                return "format_results"
            else:
                # First time, check and run the query
                return "check_query"

        def should_continue_after_run(state: MessagesState) -> Literal["generate_query", "format_results"]:
            """Decide what to do after running query"""
            messages = state["messages"]

            # Count how many times we've queried
            query_count = sum(1 for msg in messages if hasattr(msg, 'name') and msg.name == "sql_db_query")

            # After first query, go straight to formatting (no re-querying)
            if query_count >= 1:
                return "format_results"
            else:
                # This shouldn't happen, but fallback to generate_query
                return "generate_query"

        # Build the graph
        builder = StateGraph(MessagesState)

        # Add nodes
        builder.add_node("list_tables", list_tables)
        builder.add_node("call_get_schema", call_get_schema)
        builder.add_node("get_schema", self.get_schema_node)
        builder.add_node("generate_query", generate_query)
        builder.add_node("check_query", check_query)
        builder.add_node("run_query", self.run_query_node)
        builder.add_node("format_results", format_results)

        # Add edges
        builder.add_edge(START, "list_tables")
        builder.add_edge("list_tables", "call_get_schema")
        builder.add_edge("call_get_schema", "get_schema")
        builder.add_edge("get_schema", "generate_query")

        # Conditional routing after query generation
        builder.add_conditional_edges(
            "generate_query",
            should_continue_after_query,
            {
                "check_query": "check_query",
                "format_results": "format_results"
            }
        )

        builder.add_edge("check_query", "run_query")

        # After running query, go to formatting (no loop back)
        builder.add_conditional_edges(
            "run_query",
            should_continue_after_run,
            {
                "format_results": "format_results",
                "generate_query": "generate_query"  # Fallback only
            }
        )

        # Format results leads to END
        builder.add_edge("format_results", END)

        self.agent = builder.compile()


    def infer(self, question):
        """Run the agent and return the final formatted response"""
        try:
            answers = []
            step_count = 0
            max_steps = 50

            for step in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="values",
                recursion_limit=50
            ):
                step_count += 1
                answers.append(step["messages"][-1])

                last_msg = step['messages'][-1]
                step_type = last_msg.__class__.__name__

                print(f"Step {step_count}: {step_type}")

                # Check for errors in messages
                if hasattr(last_msg, 'content'):
                    content_str = str(last_msg.content).lower()
                    if 'error' in content_str or 'exception' in content_str:
                        print(f"Error detected in step {step_count}: {last_msg.content}")

                # Safety check - shouldn't happen with proper graph, but just in case
                if step_count >= max_steps:
                    print("WARNING: Max steps reached, forcing exit")
                    break

            # Get the last message content
            if not answers:
                return json.dumps({
                    "summary": "No response generated. Please try rephrasing your query.",
                    "top_properties": []
                })

            final_content = answers[-1].content

            # Clean up any markdown formatting
            import re
            final_content = re.sub(r'```json\s*|\s*```', '', final_content, flags=re.IGNORECASE)
            final_content = final_content.strip()

            # Validate it's proper JSON
            try:
                parsed = json.loads(final_content)

                # Ensure required structure
                if not isinstance(parsed, dict):
                    raise ValueError("Response is not a JSON object")

                if "summary" not in parsed:
                    parsed["summary"] = "Search completed."

                if "top_properties" not in parsed:
                    parsed["top_properties"] = []

                # Validate property structure
                if not isinstance(parsed["top_properties"], list):
                    parsed["top_properties"] = []

                return json.dumps(parsed)

            except (json.JSONDecodeError, ValueError) as e:
                print(f"JSON validation error: {e}")
                print(f"Content was: {final_content[:500]}")

                # Try to extract JSON from text
                json_match = re.search(r'\{.*"top_properties".*\}', final_content, re.DOTALL)
                if json_match:
                    try:
                        extracted = json.loads(json_match.group(0))
                        return json.dumps(extracted)
                    except:
                        pass

                # Return error JSON
                return json.dumps({
                    "summary": "Unable to parse results. The query may have failed or returned unexpected data. Please try a different search.",
                    "top_properties": []
                })

        except RecursionError as e:
            print(f"Recursion error in agent: {e}")
            return json.dumps({
                "summary": "The search became too complex and exceeded limits. Please try a simpler, more specific query.",
                "top_properties": []
            })
        except Exception as e:
            print(f"Error in agent inference: {e}")
            import traceback
            traceback.print_exc()

            return json.dumps({
                "summary": f"An error occurred while searching: {str(e)}. Please try rephrasing your query or contact support.",
                "top_properties": []
            })