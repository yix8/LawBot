from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.retrievers.multi_query import LineListOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from prompt import *
import pandas as pd
import os
import json
import  re
import time
import cohere
from dotenv import load_dotenv

load_dotenv(override=True)

class Timer:
    """Timer context manager"""

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start

    def __str__(self):
        return f"{self.duration:.1f} seconds"
    
def get_embeddings_model():
    model_map = {
        'openai': OpenAIEmbeddings(
            model = os.getenv('OPENAI_EMBEDDINGS_MODEL')
        )
    }
    return model_map.get(os.getenv('EMBEDDINGS_MODEL'))

def get_law_embeddings_model():
    return os.getenv('LAW_EMBEDDINGS_MODEL')

def get_llm_model():   
    model_map = {
        'openai': ChatOpenAI(
            model = os.getenv('OPENAI_LLM_MODEL'),
            temperature = os.getenv('TEMPERATURE'),
            max_tokens = os.getenv('MAX_TOKENS'),
        )
    }
    return model_map.get(os.getenv('LLM_MODEL'))

def get_chat_llm_model():
    model_map = {
        'openai': ChatOpenAI(
            model = os.getenv('OPENAI_CHAT_MODEL'),
            temperature = os.getenv('TEMPERATURE'),
            max_tokens = os.getenv('MAX_TOKENS'),
        )
    }
    return model_map.get(os.getenv('LLM_MODEL'))


def structured_output_parser(response_schemas):
    # Initialize the instruction text for AI to extract entities from provided text and output as JSON.
    # The JSON output should be encapsulated with "```json" at the beginning and "```" at the end.
    instruction_text = '''
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的 "```json" 和 "```"。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段：\n
    '''
    
    # Dynamically append the details of each field required in the JSON output, as per the provided schemas.
    for schema in response_schemas:
        instruction_text += f"{schema.name} 字段，表示：{schema.description}，类型为：{schema.type}\n"
    
    # Add the JSON formatting instructions at the beginning and end of the output template.
    instruction_text = "```json\n" + instruction_text + "```\n"
    
    return instruction_text

def replace_token_in_string(string, slots):
    for key, value in slots:
        string = string.replace('%'+key+'%', value)
    return string


def get_step_back_prompt():
    examples = [
        {
            "input": "警察局成员可以执行逮捕行动吗？",
            "output": "警察局成员能做些什么？"
        },
        {
            "input": "某法律顾问是在哪个国家出生的？", 
            "output": "这位法律顾问的个人历史是什么？"
        }
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是世界知识方面的专家。你的任务是将一个问题改述为一个更通用的、更容易回答的问题。以下是一些示例："""),
        few_shot_prompt,
        ("user", "{question}"),
    ])

    return prompt

def keyword_extraction(query):
        response_schemas = [
            ResponseSchema(type='list', name='title', description='法律文件名称实体'),
            ResponseSchema(type='list', name='index', description='条款编号汉字实体'),
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

        return ner_result

def refine_keyword(keyword):
    pattern = r"^([零一二三四五六七八九十百千万]+)"
    match = re.search(pattern, keyword)

    if match:
        matched_text = match.group(0)
        return (f"第{matched_text}条")
    else:
        return None
    
def get_BM25_retriever(k = 5):
    file_path = os.path.join(os.path.dirname(__file__), 'contents_repeat_index.json')
    df = pd.read_json(file_path)

    # generate documents
    documents = []
    for index, row in df.iterrows():
        documents.append(Document(page_content=row['content'], metadata={"title": row['title'], "index": row['index'], "origin": row['origin']}))

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever

def multi_query_generations(query, num = 3):
    prompt = PromptTemplate.from_template(MULTI_QUERY_TMPL)

    llm_chain = LLMChain(llm=get_llm_model(), prompt=prompt, output_parser=LineListOutputParser())

    inputs = {
        'question': query,
        'number': num
    }

    questions = llm_chain.invoke(inputs)['text']
    questions = [re.sub(r"^\d+[\).\s]", "", question).strip() for question in questions]
    questions = [question for question in questions if len(question) > 0]
    questions.append(query)
    
    return questions

def get_high_docs_score(query, docs, top_n=5, threshold=0.7):
    """get documents and scores with high relevancy
    ---
    output: a list of document text where score > threshold and a list of document text where score <= threshold
    """
    co = cohere.Client(os.getenv('COHERE_RERANK_KEY'))
    res = co.rerank(query=query, documents=docs, top_n=top_n,
                    model=os.getenv('COHERE_RERANK_MODEL'), return_documents=True)
    high_score_docs = [r.document.text for r in res.results if r.relevance_score > threshold]
    rest_docs = [r.document.text for r in res.results if r.relevance_score <= threshold]

    return high_score_docs, rest_docs

def get_rerank_documents(query, list_of_docs, min_n=3, top_n=5, threshold=0.7):
    """get documents in reranked order
    ---
    input:
    min_n: minimum documents returned
    ---
    output:
    list of reordered documents
    """
    assert(min_n <= top_n)
    high_score_docs, rest_docs = get_high_docs_score(query, list_of_docs, top_n, threshold)
    if len(high_score_docs) < min_n:
        candidate_docs = high_score_docs + rest_docs[:min_n - len(high_score_docs)]
    else:
        candidate_docs = high_score_docs
    return candidate_docs

def get_title_index(load_path = 'title_category.json'):
    load_path = os.path.join(os.path.dirname(__file__), load_path)
    with open(load_path, 'r', encoding='utf-8') as file:
        titles_list = json.load(file)

    titles = set(titles_list)
    
    documents = []
    for title in titles:
        documents.append(Document(page_content=title))

    return titles, documents

OPENAI_PRICING = {
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
    "embedding": {"hugging_face": 0, "text-embedding-ada-002": 0.0001},
}


OPENAI_MODEL_CONTEXT_LENGTH = {
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
}