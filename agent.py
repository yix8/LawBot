from utils import *
from config import *
from prompt import *

import os
from langchain.chains import LLMChain, LLMRequestsChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain import hub
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

class Agent():
    def __init__(self):
        self.vdb = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__), './data/db'), 
            embedding_function = get_embeddings_model()
        )

        self.embedding_function = SentenceTransformerEmbeddings(model_name=get_law_embeddings_model())

        self.lawdb = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__), './data/laws_db_1000'), 
            embedding_function = self.embedding_function
        )

    def generic_func(self, x, query):
        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
        llm_chain = LLMChain(
            llm = get_llm_model(), 
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        return llm_chain.invoke(query)['text']
    
    def retrival_func(self, x, query):
        # Retrieve and filter documents based on the given query.
        documents = sorted(self.vdb.similarity_search_with_relevance_scores(query, k=5), key=lambda doc: doc[1], reverse=True)

        high_score_docs  = [doc[0].page_content for doc in documents if doc[1]>0.7]

        if len(high_score_docs) < 3:
            fill_docs = [doc[0].page_content for doc in documents[len(high_score_docs):3]]
            query_result = high_score_docs + fill_docs
        else:
            query_result = high_score_docs

        # Fill in the prompt template and summarize the answer.
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        retrival_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else 'Knowledge not found.'
        }
        return retrival_chain.invoke(inputs)['text']

    def retrival_law_func(self, x, query):
        # Retrieve and filter documents based on the given query.
        documents = sorted(self.lawdb.similarity_search_with_relevance_scores(query, k=5), key=lambda doc: doc[1], reverse=True)

        high_score_docs  = [doc[0].page_content for doc in documents]

        if len(high_score_docs) < 3:
            fill_docs = [doc[0].page_content for doc in documents[len(high_score_docs):3]]
            query_result = high_score_docs + fill_docs
        else:
            query_result = high_score_docs

        # Fill in the prompt template and summarize the answer.
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        retrival_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else 'Knowledge not found.'
        }
        return retrival_chain.invoke(inputs)['text']

    def search_func(self, query):
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        llm_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        llm_request_chain = LLMRequestsChain(
            llm_chain = llm_chain,
            requests_key = 'query_result'
        )
        inputs = {
            'query': query,
            'url': 'https://www.google.com/search?q='+query.replace(' ', '+')
        }
        return llm_request_chain.invoke(inputs)['output']
    
    def query(self, query):
        tools = [
            Tool.from_function(
                name = 'generic_func',
                func = lambda x: self.generic_func(x, query),
                description = 'Answers general knowledge questions such as greetings or inquiries about identity.',
            ),
            Tool.from_function(
                name = 'retrival_func',
                func = lambda x: self.retrival_func(x, query),
                description = 'Responds to questions related to developer background and similar topics such as team members or contactions.',
            ),
            Tool(
                name = 'retrival_law_func',
                func = lambda x: self.retrival_law_func(x, query),
                description = 'Handling consultations on legal matters, including criminal law, constitutional law, civil law, social law, economic law, administrative law, and local regulations among other legal issues.',
            ),
            Tool(
                name = 'search_func',
                func = self.search_func,
                description = 'Utilizes search engines to answer general questions when other tools cannot provide a correct response.',
            ),
        ]

        prompt = hub.pull('hwchase17/react-chat')
        prompt.template = 'Please answer the question in English! The Final Answer must respect the outcome of Obversion, without altering its meaning.\n\n' + prompt.template
        agent = create_react_agent(llm=get_llm_model(), tools=tools, prompt=prompt)
        memory = ConversationBufferMemory(memory_key='chat_history')
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent = agent, 
            tools = tools, 
            memory = memory, 
            handle_parsing_errors = True,
            verbose = os.getenv('VERBOSE')
        )
        return agent_executor.invoke({"input": query})['output']
    
if __name__ == '__main__':
    agent = Agent()
    #print(agent.generic_func('Who are you? What is linear regression?'))
    print(agent.query('Introduce the team members'))
    
    #print(agent.search_func('what is the game gbf?'))
    #print(agent.query("What is the background of the team?"))
    # print(agent.query("Which is the biggest animial in the world?"))
    # print(agent.query("What is the current time, use google?"))
    # print(agent.query("在中国, 结婚的法定年龄是多少岁, 依据是什么?"))
    # print(agent.query("我是一个中国妇女, 我有一个6个月的小孩, 我老公想跟我离婚, 根据民法典, 我能拒绝离婚吗?"))
    # print(agent.query("我是一个中国妇女, 我有一个6个月的小孩, 我老公想跟我离婚, 根据民法典第一千零八十二条, 他能提出离婚吗?"))
