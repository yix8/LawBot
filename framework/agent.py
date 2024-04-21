from utils import *
from prompt import *
import os
from langchain.chains import LLMChain, LLMRequestsChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.retrievers import EnsembleRetriever

class Agent():
    def __init__(self, k = 5, suffix = 'meta', use_query = True):
        self.vdb = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__), './data/db'), 
            embedding_function = get_embeddings_model()
        )
        
        self.embedding_function = SentenceTransformerEmbeddings(model_name=get_law_embeddings_model())

        self.lawdb = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__), f'./data/laws_db_{suffix}'), 
            embedding_function = self.embedding_function
        )

        self.sparse_retriver = get_BM25_retriever(k)

        self.law_retriver = EnsembleRetriever(
            retrievers=[self.lawdb.as_retriever(search_kwargs={"k": k}), self.sparse_retriver], weights=[0.6, 0.4]
        )

        title_indexs, title_docs = get_title_index()
        self.title_indexs = title_indexs
        self.title_db = Chroma.from_documents(title_docs, get_embeddings_model())

        self.use_query = use_query

        if use_query:
            model_path = os.path.join(os.path.dirname(__file__), './finetune_data/BGE_S_L')
            self.query_db = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__), './finetune_data/data/query_db_finetune'), 
            embedding_function = SentenceTransformerEmbeddings(model_name=model_path)
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

        docs  = [doc[0].page_content for doc in documents]
        query_result = get_rerank_documents(query, docs, min_n=5, top_n=7, threshold=0.7)

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
        # general multiple similar queries
        queries = multi_query_generations(query, 3)
        query_result = []
        # Retrieve and filter documents based on the given query.
        for per_query in queries:
            documents = self.law_retriver.get_relevant_documents(per_query)
            query_result = set(query_result) | set([doc.metadata['origin'] for doc in documents])

        # use key words to retrieve again
        indexs = keyword_extraction(query)['index']
        titles = keyword_extraction(query)['title']

        for index in indexs:
            keyword = refine_keyword(index)
            if keyword:
                keyword_documents = sorted(self.lawdb.similarity_search_with_relevance_scores(query, k=3, filter={"index": keyword}), key=lambda doc: doc[1], reverse=True)
                query_result = set(query_result) | set([doc[0].metadata['origin'] for doc in keyword_documents])
        
        for title in titles:
            keyword = title
            if keyword not in self.title_indexs:
                keyword = self.title_db.similarity_search(query, k=1)[0].page_content
            keyword_documents = sorted(self.lawdb.similarity_search_with_relevance_scores(query, k=3, filter={"title": keyword}), key=lambda doc: doc[1], reverse=True)
            query_result = set(query_result) | set([doc[0].metadata['origin'] for doc in keyword_documents])

        if self.use_query:
            query_documents = sorted(self.query_db.similarity_search_with_relevance_scores(query, k=3), key=lambda doc: doc[1], reverse=True)
            query_result = set(query_result) | set([doc[0].page_content for doc in query_documents])

        query_result = list(query_result)
        query_result = get_rerank_documents(query, query_result, min_n=5, top_n=5, threshold=0.7)

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
                description = 'Answers general knowledge questions which is not related to law.',
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
        prompt.template = 'Please answer the question in Chinese! The Final Answer must respect the outcome of Obversion, without altering its meaning.\n\n' + prompt.template
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