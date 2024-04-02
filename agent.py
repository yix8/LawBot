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

class Agent():
    def __init__(self):
        self.vdb = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__), './data/db'), 
            embedding_function = get_embeddings_model()
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

    def graph_func(self, x, query):
        # Entity information extraction
        response_schemas = [
            ResponseSchema(type='list', name='disease', description='疾病名称实体'),
            ResponseSchema(type='list', name='symptom', description='疾病症状实体'),
            ResponseSchema(type='list', name='drug', description='药品名称实体'),
        ]

        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        format_instructions = structured_output_parser(response_schemas)

        ner_prompt = PromptTemplate(
            template = NER_PROMPT_TPL,
            partial_variables = {'format_instructions': format_instructions},
            input_variables = ['query']
        )

        ner_chain = LLMChain(
            llm = get_llm_model(),
            prompt = ner_prompt,
            verbose = os.getenv('VERBOSE')
        )

        result = ner_chain.invoke({
            'query': query
        })['text']
        
        ner_result = output_parser.parse(result)

        graph_templates = []
        for key, template in GRAPH_TEMPLATE.items():
            slot = template['slots'][0]
            slot_values = ner_result[slot]
            for value in slot_values:
                graph_templates.append({
                    'question': replace_token_in_string(template['question'], [[slot, value]]),
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                })
        if not graph_templates:
            return 
        
        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates
        ]
        db = FAISS.from_documents(graph_documents, get_embeddings_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)
        print(graph_documents_filter)

        query_result = []
        neo4j_conn = get_neo4j_conn()
        for document in graph_documents_filter:
            question = document[0].page_content
            cypher = document[0].metadata['cypher']
            answer = document[0].metadata['answer']
            try:
                result = neo4j_conn.run(cypher).data()
                if result and any(value for value in result[0].values()):
                    answer_str = replace_token_in_string(answer, list(result[0].items()))
                    query_result.append(f'Question: {question}\nAnswer:{answer_str}')
            except:
                pass
        print(query_result)
            
        prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
        graph_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else 'Empty.'
        }
        return graph_chain.invoke(inputs)['text']
    
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
                name = 'graph_func',
                func = lambda x: self.graph_func(x, query),
                description = 'Addresses medical-related inquiries, including diseases, symptoms, and medications.',
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
    #print(agent.retrival_func('Introduce the team members'))
    
    # print(agent.graph_func('鼻炎怎么治疗？'))
    #print(agent.search_func('what is the game gbf?'))
    #print(agent.query("What is the background of the team?"))
    # print(agent.query("Which is the biggest animial in the world?"))
    print(agent.query("What is the current time, use google?"))
