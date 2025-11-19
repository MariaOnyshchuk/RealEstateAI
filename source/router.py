##########
from langchain.agents import create_agent

class Router:

    AVAILABLE_MODELS = ["family", "investor", "young_professional"]

    def __init__(self, llm):

        self.agent = create_agent(llm, 
            system_prompt="""You are question based-router. You will receive message and based on it
            you should return how confident are you that this message corresponds to "family", "investor" or "young_professional" categories.
            Example: "Where would you recommend to live young athlete?" - "young_professional: 0.99; investor: 0.3; family: 0.02". And only this
            without any justification.
            """)


    def infer(self, question):

        res = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )

        # print(f'\n\nres: {}\n\n')
        return res['messages'][-1].content
    
    @staticmethod
    def analyze_router_output(router_out):

        confidences = {"family": 0, "investor": 0, "young_professional": 0}
        spitted = router_out.split('; ')

        for splitt in spitted:
            name, conf = splitt.split(': ')

            if name not in confidences:
                continue

            confidences[name] = float(conf)

        return confidences
    
    def select_models(self, question):

        models = []
        output = self.infer(question)
        print(output)

        if (output[0], output[-1]) == ('"', '"'):
            output = output[1:-1]

        confidences = self.analyze_router_output(output)
        print(f"{confidences=}")

        for model, confidence in confidences.items():
            if confidence >= 0.37:
                models.append(model)

        if len(models) == 0:
            models.append(max(confidences.items(), key=lambda x: x[1]))

        return output, models