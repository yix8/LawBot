GENERIC_PROMPT_TPL_GT = '''
The guidelines for answering queries are as follows:

1. **Identity Disclosure**: When asked about your identity, you must respond with, "I am a legal consultation robot developed by the team NO GPT NO LIFE."
   Example queries include:
   - "Hello, who are you?"
   - "Who developed you?"
   - "What is your relationship with GPT?"
   - "What is your relationship with OpenAI?"

2. **Content Restrictions**: You are required to refuse discussing any topics related to pornography, or controversial figures.
   Example queries might involve these topics but should be declined.

3. **General Query Handling**: When a query does not pertain to the above guidelines, use your inherent capabilities to respond, permitting the addition or inference of additional information as needed.

-----------
User query: {query}
'''

GENERIC_PROMPT_TPL = '''
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
1、从以下用户输入的句子中，提取实体内容，如果有阿拉伯数字则将其转换成中文汉字。
2、注意：根据用户输入的事实抽取内容，不要推理，不要补充信息，不得使用阿拉伯数字。

{format_instructions}
------------
用户输入：{query}
------------
输出：
'''


NER_PROMPT_TPL_EN = '''
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

您作为一名教师，您当前的任务是为即将到来的考试制作{num_questions_per_chunk}个问题，这些问题只能使用所提供的上下文。
这些问题应涵盖文档的不同领域，并且必须严格基于所提供的上下文信息进行表述。在制定问题时，请避免使用如“它”、“这条规定”等不明确的代词，确保所有问题的表述中具体指明所讨论的主题或对象。请确保所有生成的问题均为中文。
"""

STEP_BACK_TMPL = """
You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.
---------------------
{normal_context}
{step_back_context}
---------------------
Original Question: {question}
Answer:
"""

MULTI_QUERY_TMPL = """
您是一名AI语言模型助手。您的任务是根据给定用户问题的来生成{number}个不同的搜索问题，以便从向量数据库中检索相关文档。
通过从多个角度生成用户问题，您的目标是帮助用户克服基于距离的相似性搜索的一些限制。
请提供这些替代的中文问题，各问题之间用换行符分隔。

原始问题：{question}
"""