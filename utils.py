from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from py2neo import Graph
from config import *
import cohere

import os
from dotenv import load_dotenv
load_dotenv()


def get_embeddings_model():
    model_map = {
        'openai': OpenAIEmbeddings(
            model = os.getenv('OPENAI_EMBEDDINGS_MODEL')
        )
    }
    return model_map.get(os.getenv('EMBEDDINGS_MODEL'))

def get_llm_model():
    model_map = {
        'openai': ChatOpenAI(
            model = os.getenv('OPENAI_LLM_MODEL'),
            temperature = os.getenv('TEMPERATURE'),
            max_tokens = os.getenv('MAX_TOKENS'),
        )
    }
    return model_map.get(os.getenv('LLM_MODEL'))


def structured_output_parser(response_schemas):
    # Initialize the instruction text for AI to extract entities from provided text and output as JSON.
    # The JSON output should be encapsulated with "```json" at the beginning and "```" at the end.
    instruction_text = '''
    Please extract entity information from the text below and format it as JSON.
    The JSON output should include the following fields, as described. Each field is mandatory in the output JSON:\n
    '''
    
    # Dynamically append the details of each field required in the JSON output, as per the provided schemas.
    for schema in response_schemas:
        instruction_text += f"{schema.name} field, described as: {schema.description}, with a type of: {schema.type}\n"
    
    # Add the JSON formatting instructions at the beginning and end of the output template.
    instruction_text = "```json\n" + instruction_text + "```\n"
    
    return instruction_text


def replace_token_in_string(string, slots):
    for key, value in slots:
        string = string.replace('%'+key+'%', value)
    return string

def get_neo4j_conn():
    return Graph(
        os.getenv('NEO4J_URI'), 
        auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )

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