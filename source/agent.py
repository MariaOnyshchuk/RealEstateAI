##########
from langchain_community.utilities import SQLDatabase

from typing import Literal

from langgraph.prebuilt import ToolNode
from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

class Agent:

    def __init__(self, llm, agent_type):

        db = SQLDatabase.from_uri("sqlite:///geeks2.db")

        print(f"Dialect: {db.dialect}")
        print(f"Available tables: {db.get_usable_table_names()}")
        print(f'Sample output: {db.run("SELECT * FROM DB LIMIT 5;")}')

        from langchain_community.agent_toolkits import SQLDatabaseToolkit

        self.llm = llm
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)

        tools = toolkit.get_tools()

        for tool in tools:
            print(f"{tool.name}: {tool.description}\n")

        self.tools = tools
        self.db = db

        self.agent_type = agent_type

        self.get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
        self.get_schema_node = ToolNode([self.get_schema_tool], name="get_schema")

        self.run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
        self.run_query_node = ToolNode([self.run_query_tool], name="run_query")

        self.generate_query_system_prompt = """
        You are an agent designed to interact with a SQL database. You have a database with different 
        real estate listings, and you should target offers that are best suitable for {agent_type}.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query, analyse the result based on your target_offers {agent_type} and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        """.format(
            dialect=db.dialect,
            top_k=5,
            agent_type=agent_type
        )

        self.check_query_system_prompt = """
        You are a SQL expert with a strong attention to detail.
        Double check the {dialect} query for common mistakes, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins

        If there are any of the above mistakes, rewrite the query. If there are no mistakes,
        just reproduce the original query.

        You will call the appropriate tool to execute the query after running this check.
        """.format(dialect=db.dialect)

        # Example: create a predetermined tool call
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


        # Example: force a model to create a tool call
        def call_get_schema(state: MessagesState):
            # Note that LangChain enforces that all models accept `tool_choice="any"`
            # as well as `tool_choice=<string name of tool>`.
            llm_with_tools = self.llm.bind_tools([self.get_schema_tool], tool_choice="any")
            response = llm_with_tools.invoke(state["messages"])

            return {"messages": [response]}


        def generate_query(state: MessagesState):
            system_message = {
                "role": "system",
                "content": self.generate_query_system_prompt,
            }
            # We do not force a tool call here, to allow the model to
            # respond naturally when it obtains the solution.
            llm_with_tools = self.llm.bind_tools([self.run_query_tool])
            response = llm_with_tools.invoke([system_message] + state["messages"])

            return {"messages": [response]}


        def check_query(state: MessagesState):
            system_message = {
                "role": "system",
                "content": self.check_query_system_prompt,
            }

            # Generate an artificial user message to check
            tool_call = state["messages"][-1].tool_calls[0]
            user_message = {"role": "user", "content": tool_call["args"]["query"]}
            llm_with_tools = self.llm.bind_tools([self.run_query_tool], tool_choice="any")
            response = llm_with_tools.invoke([system_message, user_message])
            response.id = state["messages"][-1].id

            return {"messages": [response]}

        def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
            messages = state["messages"]
            last_message = messages[-1]
            if not last_message.tool_calls:
                return END
            else:
                return "check_query"

        builder = StateGraph(MessagesState)
        builder.add_node(list_tables)
        builder.add_node(call_get_schema)
        builder.add_node(self.get_schema_node, "get_schema")
        builder.add_node(generate_query)
        builder.add_node(check_query)
        builder.add_node(self.run_query_node, "run_query")

        builder.add_edge(START, "list_tables")
        builder.add_edge("list_tables", "call_get_schema")
        builder.add_edge("call_get_schema", "get_schema")
        builder.add_edge("get_schema", "generate_query")
        builder.add_conditional_edges(
            "generate_query",
            should_continue,
        )
        builder.add_edge("check_query", "run_query")
        builder.add_edge("run_query", "generate_query")

        self.agent = builder.compile()

    def infer(self, question):

        answers = []
        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
            recursion_limit=100
        ):
            # step["messages"][-1].pretty_print()
            answers.append(step["messages"][-1])

        return answers[-1].content