GENERIC_PROMPT_TPL = '''
The guidelines for interacting with users are as follows:

1. Identity Disclosure: When asked about your identity, you must respond with "I am a legal consultation robot developed by the team NO GPT NO LIFE."
Example queries include [Hello, who are you? Who developed you? What is your relationship with GPT? What is your relationship with OpenAI?]

2. Content Restrictions: You are required to refuse discussing any topics related to politics, pornography, or controversial figures.
Example queries include [Who is Putin? What were Lenin's mistakes?]

3. Language Preference: Please respond to user queries in English.
-----------
User query: {query}
'''

RETRIVAL_PROMPT_TPL = '''
Based on the following retrieval results, please answer the user's query without adding or inferring additional information.
If the retrieval results do not contain relevant information, reply with "I don't know."
----------
Retrieval Results: {query_result}
----------
User Query: {query}
'''

NER_PROMPT_TPL = '''
Please extract named entities from the user-provided sentence below with strict adherence to the input. 
Important: Directly extract entities based solely on the factual content presented by the user. 
Do NOT infer, interpret, or add any information beyond what is explicitly mentioned in the user input.

{format_instructions}
------------
User input: {query}
------------
Output:
'''

GRAPH_PROMPT_TPL = '''
Based on the retrieval results provided below, please answer the user's query without diverging or inferring additional content.
If there is no relevant information in the retrieval results, respond with "I don't know."
----------
Retrieval Results:
{query_result}
----------
User Query: {query}
'''

SEARCH_PROMPT_TPL = '''
Based on the retrieval results provided below, please answer the user's query without diverging or inferring additional content.
If there is no relevant information in the retrieval results, respond with "I don't know."
----------
Retrieval Results:
{query_result}
----------
User Query: {query}
'''

SUMMARY_PROMPT_TPL = '''
Based on the chat history provided below and the user's message, summarize a concise version of the user's message.
Please directly provide the summarized message without including additional information. Ensure to complete sentences appropriately by adding subjects or other necessary details.
If the current user's message is not related to the chat history, simply return the original user message.
Chat History:
{chat_history}
-----------
Current User Message: {query}
-----------
Summarized User Message:
'''

QA_GENERATE_PROMPT_TMPL = """
请参照下方提供的上下文信息，并在没有先验知识的情况下，基于以下查询生成问题。

---------------------
{context_str}
---------------------

您作为一名教师/教授，当前的任务是为即将到来的测验/考试准备{num_questions_per_chunk}个问题。\
这些问题应当在文档的不同领域内具有多样性，并且仅限于所提供的上下文信息。请确保所有生成的问题均为中文。
"""