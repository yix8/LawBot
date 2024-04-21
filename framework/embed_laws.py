import pandas as pd
from langchain.schema.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma

from utils import *

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), 'contents_repeat_index.json')
    df = pd.read_json(file_path)
    
    # generate documents
    documents = []
    for index, row in df.iterrows():
        documents.append(Document(page_content=row['content'], metadata={"title": row['title'], "index": row['index'], "origin": row['origin']}))

    print(f'Embedding and store nodes.')
    embedding_function = SentenceTransformerEmbeddings(model_name=get_law_embeddings_model())

    # embedding and store
    if documents:
        vdb = Chroma.from_documents(
            documents = documents, 
            embedding = embedding_function,
            persist_directory = os.path.join(os.path.dirname(__file__), './data/laws_db_meta/')
        )
        vdb.persist()