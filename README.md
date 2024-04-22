# LawBot: Enhancing LLMs with RAG for Legal Precision
Group Project of COMP0087: Statistical Natural Language Processing

## Introduction
LawBot is a simple yet effective framework tailored to specialized legal domains, operational without training. Utilizing Chinese legal and regulatory documents as the knowledge base, LawBot enhances the breadth of retrieval through multi-query generation and hybrid search strategies. It increases precision with metadata filtering and confirms the plausibility of knowledge through context-based reranking. Remarkably, all these procedures are conducted via zero-shot prompting, making LawBot broadly applicable even when LLMs are accessible only through a black-box API.

![LawBot Pipeline](./imgs/pipeline.png) 

## Installation

Follow these steps to set up the LawBot environment on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/yix8/LawBot.git
   ```
2. Navigate to the LawBot directory:
   ```bash
   cd LawBot
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Build the general vector store:
   ```bash
   python framework/embed_laws.py
   ```
5. Build the specific vector store:
   ```bash
   python framework/finetune_data/embed_query.py
   ```

## Configuration

Set up the necessary API keys in the `.env` file located in the `framework` folder:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `COHERE_RERANK_KEY`: Your Cohere rerank API key.
- `LANGCHAIN_API_KEY`: Your LangChain API key.
- `LANGCHAIN_PROJECT`: Your LangChain project identifier.

## Running LawBot
To interact with the model via a web interface:
   ```bash
   python framework/App.py
   ```

![Interface](./imgs/Article_Recitation.png) 

## China Law Query Synthetic
We also proposed an open-source QA dataset, the [Chinese Legal Question Answering dataset (CLQS)](CLQS/qa_full.json) which can be utilized as an instruction dataset.
