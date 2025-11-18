import os
from langchain_classic.memory import ConversationSummaryBufferMemory
# from langchain.memory.buffer import ConversationBufferMemory
# from langchain.chains import ConversationChain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_aws import ChatBedrockConverse

from dotenv import load_dotenv

load_dotenv()
ACCESS_ID = os.getenv('access_id'.upper())
ACCESS_KEY = os.getenv('access_key'.upper())

# print(type(ACCESS_ID))

os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = ACCESS_KEY

# Shared AI connection
_bedrock_llm = None

def get_bedrock_client():
    """Create AWS Bedrock AI connection"""
    global _bedrock_llm
    if _bedrock_llm is None:
        # model_id = os.getenv('BEDROCK_MODEL_ID', 'amazon.nova-pro-v1:0')
        model_id = os.getenv('BEDROCK_MODEL_ID', 'amazon.nova-lite-v1:0')
        _bedrock_llm = ChatBedrockConverse(
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            model=model_id,
            temperature=0.1,
            max_tokens=1000
        )
    return _bedrock_llm

def create_chat_memory():
    """Create conversation memory"""
    return ConversationSummaryBufferMemory(
        llm=get_bedrock_client(), 
        max_token_limit=2000
    )

def get_ai_response(user_input, store_new):
    """Get AI response with memory"""
    # conversation_chain = ConversationChain(
    #     llm=get_bedrock_client(), 
    #     memory=chat_memory, 
    #     verbose=True
    # )
    # return conversation_chain.invoke(user_input)['response']
    global store
    store = store_new

    # RunnableWithMessageHistory()
    llm = get_bedrock_client()

    conversation_chain = RunnableWithMessageHistory(llm, get_session_history=get_session_history)
    res = conversation_chain.invoke(
        user_input,
        config={"configurable": {"session_id": "1"}},
    )

    # print(f'\n\nres: {res.__dict__}\n\n')

    return res.content


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    memory = ConversationSummaryBufferMemory(
        chat_memory=store[session_id],
        llm=get_bedrock_client(), 
        max_token_limit=2000
    )

    assert len(memory.memory_variables) == 1
    key = memory.memory_variables[0]
    messages = memory.load_memory_variables({})[key]
    # print(repr(messages))

    translated_messages = []
    for message in messages.split('\n'):
        if message[:3] == 'AI:':
            translated_messages.append(AIMessage(message[4:]))
        else:
            assert message[:6] == 'Human:'
            translated_messages.append(HumanMessage(message[7:]))

    store[session_id] = InMemoryChatMessageHistory(messages=translated_messages)
    return store[session_id]