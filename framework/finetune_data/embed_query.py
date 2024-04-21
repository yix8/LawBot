import pandas as pd
from langchain.schema.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
import os

parent_path = os.path.dirname(__file__)
model_path = os.path.join(parent_path, 'BGE_S_L')

if __name__ == '__main__':
    df = pd.read_json(os.path.join(parent_path, 'qa_subset_filter.json'))
    
    # generate documents
    documents = []
    for index, row in df.iterrows():
        documents.append(Document(page_content=row['content']))

    print(f'Embedding and store nodes.')
    embedding_function = SentenceTransformerEmbeddings(model_name=model_path)

    # embedding and store
    if documents:
        vdb = Chroma.from_documents(
            documents = documents, 
            embedding = embedding_function,
            persist_directory = os.path.join(os.path.dirname(__file__), './data/query_db_finetune/')
        )
        vdb.persist()