import pandas as pd
from langchain.schema.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
import os

parent_path = os.path.dirname(__file__)
model_path = os.path.join(parent_path, 'BGE_S_L')

if __name__ == '__main__':
    df = pd.read_json(os.path.join(parent_path, 'qa_subset_filter.json'))
    

    print(f'Loading db.')
    embedding_function = SentenceTransformerEmbeddings(model_name='BAAI/bge-small-zh-v1.5')

    db = Chroma(
                persist_directory = os.path.join(parent_path, './data/query_db'), 
                embedding_function = embedding_function
            )
    
    # evaluate hit rate
    k = 5
    total = 0
    count = 0
    for index, row in df.iterrows():
        content = row['content']
        queries = row['questions']
        print(f"Evaluating {index}.")
        for query in queries:
            retrivals = sorted(db.similarity_search_with_relevance_scores(query, k=k), key=lambda doc: doc[1], reverse=True)
            docs  = [doc[0].page_content for doc in retrivals]
            if content in docs:
                count += 1
            total += 1
    print(total)
    print(f"Hit-rate with top k = {k}: {count/total}")

