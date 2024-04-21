from utils import *

import os
import pandas as pd
from glob import glob
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def doc2vec(saving_path = 'nodes.json'):
    # Initialize a RecursiveCharacterTextSplitter with specific chunk size and overlap.
    # This splitter will be used to divide the documents into smaller, more manageable pieces.
    text_splitter = RecursiveCharacterTextSplitter(
    separators = [''],
    chunk_size = 1000,
    chunk_overlap = 100
    )

    dir_path = os.path.join(os.path.dirname(__file__), './chinese_laws')
    documents = []
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)  
            loader = None  
            
            if file_name.endswith('.md'):
                loader = UnstructuredMarkdownLoader(file_path)

            print(f'Loading {file_name}.')
            if loader:
                documents += loader.load_and_split(text_splitter)
    
    print(len(documents))

    print(f"Saving content to {saving_path}.")
    nodes = [document.page_content for document in documents]

    df = pd.DataFrame(nodes, columns=['content'])
    json_str = df.to_json(orient='records', lines=False, force_ascii=False, indent=4)

    with open(saving_path, 'w', encoding='utf-8') as file:
        file.write(json_str)
    
    print(f'Embedding and store nodes.')
    embedding_function = SentenceTransformerEmbeddings(model_name=get_law_embeddings_model())

    # embedding and store
    if documents:
        vdb = Chroma.from_documents(
            documents = documents, 
            embedding = embedding_function,
            persist_directory = os.path.join(os.path.dirname(__file__), './data/laws_db_1000/')
        )
        vdb.persist()

if __name__ == '__main__':
    doc2vec('contents_1000.json')