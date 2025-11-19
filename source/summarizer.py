##########
from langchain.agents import create_agent

class Summarizer:

    def __init__(self, llm):

        self.agent = create_agent(llm, 
            system_prompt="""You are summarizer aget. You will receive message with message sender name and based on it
            you should summarize them. Possible senders are "family", "investor" and "young_professional".
            """)


    def infer(self, question):

        res = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )

        # print(f'\n\nres: {}\n\n')
        return res['messages'][-1].content