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
    "investor": ["roi", "annual_rent", "maintenance", "annual_cost", "net_rental_yield"],
    "young_professional": ["night_club", "restaurant", "gym", "library", "commute_minutes"]
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
        additional_fields = AGENT_EXTRA_FIELDS[self.agent_type]
        # additional = ': appropriate format'.join(additional_fields)

        self.generate_query_system_prompt = GENERATE_QUERY.format(
            agent_type=self.agent_type,
            dialect=self.dialect,
            table_name=self.table_name,
            top_k=self.top_k,
            additional_fields = additional_fields
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
            system_message = {
                "role": "system",
                "content": self.generate_query_system_prompt,
            }

            llm_with_tools = self.llm.bind_tools([self.run_query_tool])
            response = llm_with_tools.invoke([system_message] + state["messages"])

            return {"messages": [response]}

        def check_query(state: MessagesState):
            """
            Review the generated SQL query for correctness, block dangerous operations,
            and prepare it to run safely.
            """
            last_msg = state["messages"][-1]
            if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
                return {"messages": [AIMessage(content="No query found to check.")]}

            query = last_msg.tool_calls[0]["args"]["query"]
            query_lower = query.lower()

            # query = last_msg.tool_calls[0]["args"]["query"]
            print("Generated SQL Query:\n", query)

            dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'truncate', 'alter']
            if any(keyword in query_lower for keyword in dangerous_keywords):
                return {"messages": [AIMessage(content="Query contains dangerous operations and was blocked.")]}

            if 'select' not in query_lower:
                print(f"WARNING: Query doesn't contain SELECT: {query}")

            table_check = f'from {self.table_name.lower()}'
            if table_check not in query_lower and f'from `{self.table_name.lower()}`' not in query_lower:
                print(f"WARNING: Query might have wrong table name: {query}")

            system_message = {"role": "system", "content": self.check_query_system_prompt}
            user_message = {"role": "user", "content": query}

            llm_with_tools = self.llm.bind_tools([self.run_query_tool], tool_choice="any")
            response = llm_with_tools.invoke([system_message, user_message])

            return {"messages": [response]}

        def format_query_results(state: MessagesState):
            messages = state["messages"]

            sql_results = None
            for msg in reversed(messages):
                if hasattr(msg, '__class__') and msg.__class__.__name__ == 'ToolMessage':
                    sql_results = msg.content
                    break

            if not sql_results or sql_results.strip() in ["[]", "", "None"]:
                return {"messages": [AIMessage(content=json.dumps({
                    "summary": "No properties found matching your criteria.",
                    "top_properties": []
                }))]}

            agent_instructions = AGENT_PROMPTS.get(self.agent_type, "You are a real estate agent.")
            extra_fields = ", ".join(f'"{f}": int or null' for f in AGENT_EXTRA_FIELDS.get(self.agent_type, []))

            format_prompt_full = FORMAT_PROMPT.format(
                agent_instructions=agent_instructions,
                sql_results=sql_results,
                extra_fields=extra_fields
            )

            print('Formatted query prompt:\n', format_prompt_full)

            response = self.llm.invoke([{"role": "user", "content": format_prompt_full}])
            content = getattr(response, 'content', str(response))

            return {"messages": [AIMessage(content=content)]}


        def format_results(state: MessagesState):
            messages = state["messages"]
            query_result = None

            for msg in reversed(messages):
                if hasattr(msg, "content"):
                    if isinstance(msg.content, str):
                        query_result = msg.content
                        break
                    elif isinstance(msg.content, list):
                        for item in msg.content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                query_result = item.get('text', '')
                                break
                        if query_result:
                            break

            if not query_result:
                result_json = QueryResult(summary="No results could be formatted.", top_properties=[])
                return {"messages": [AIMessage(content=result_json.model_dump_json())]}

            query_result_clean = re.sub(r'```json\s*|\s*```', '', query_result, flags=re.IGNORECASE).strip()

            try:
                print('Query Result JSON:\n', query_result_clean)
                parsed_dict = json.loads(query_result_clean)

                numeric_fields = {"list_price", "beds", "full_baths", "sqft", "roi"}
                numeric_fields.update(AGENT_EXTRA_FIELDS.get(self.agent_type, []))

                for prop in parsed_dict.get("top_properties", []):
                    for key in numeric_fields:
                        if key in prop:
                            val = prop[key]
                            if val is None or val == "":
                                prop[key] = None
                            else:
                                try:
                                    prop[key] = float(val)
                                except (ValueError, TypeError):
                                    prop[key] = None

                    if "property_url" not in prop or not prop["property_url"]:
                        prop["property_url"] = ""

                result_json = QueryResult(**parsed_dict)
                return {"messages": [AIMessage(content=result_json.model_dump_json())]}

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                result_json = QueryResult(
                    summary="Error formatting results. Please try a different search.",
                    top_properties=[]
                )
                return {"messages": [AIMessage(content=result_json.model_dump_json())]}


        builder = StateGraph(MessagesState)

        builder.add_node("list_tables", list_tables)
        builder.add_node("call_get_schema", call_get_schema)
        builder.add_node("get_schema", self.get_schema_node)
        builder.add_node("generate_query", generate_query)
        builder.add_node("check_query", check_query)
        builder.add_node("run_query", self.run_query_node)
        builder.add_node("format_query_results", format_query_results)  # NEW NODE
        builder.add_node("format_results", format_results)

        builder.add_edge(START, "list_tables")
        builder.add_edge("list_tables", "call_get_schema")
        builder.add_edge("call_get_schema", "get_schema")
        builder.add_edge("get_schema", "generate_query")
        builder.add_edge("generate_query", "check_query")
        builder.add_edge("check_query", "run_query")
        builder.add_edge("run_query", "format_query_results")  # NEW EDGE
        builder.add_edge("format_query_results", "format_results")  # NEW EDGE
        builder.add_edge("format_results", END)

        self.agent = builder.compile()


    def infer(self, prompt: str) -> dict:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        content = result["messages"][-1].content.strip()
        content = re.sub(r"```json|```", "", content).strip()

        return json.loads(content)

