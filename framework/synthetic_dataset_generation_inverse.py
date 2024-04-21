from prompt import *
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import pandas as pd
import os
import re
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv(override=True)

def get_llm_model():
    model_map = {
        'openai': ChatOpenAI(
            model = os.getenv('OPENAI_LLM_MODEL'),
            temperature = os.getenv('TEMPERATURE'),
            max_tokens = os.getenv('MAX_TOKENS'),
            openai_api_base = os.getenv('OPENAI_BASE_URL')
        )
    }
    return model_map.get(os.getenv('LLM_MODEL'))

class AutoCahce:
    def __init__(self, filepath):
        self.cache = {}
        self.filepath = filepath
        self.df_annotation = None
        self.load_cache()
        
    def load_cache(self):
        try:
            df = pd.read_json(self.filepath)
            print(f"Loading cache from {self.filepath}.")
            self.df_annotation = df
            for index, row in df.iterrows():
                key = (row['content'])
                self.cache[key] = True
        except ValueError:
            print('Cache path is incorrect or not a JSON file, start using a new cache.')
            pass
        except FileNotFoundError:
            print('Cache file cannot be found, start using a new cache.')
            pass
        
    def get_value(self, content: str):
        return self.df_annotation[(self.df_annotation['content'] == content)]
        
    def check_key(self, content: str):
        cache_key = (content)
        return cache_key in self.cache
    
    def save_response(self, new_df: pd.DataFrame):
        try:
            df = pd.read_json(self.filepath)
        except (FileNotFoundError, ValueError):
            df = pd.DataFrame(columns=new_df.columns)
        
        df = pd.concat([df, new_df], ignore_index=True)
        self.df = df

        json_str = df.to_json(orient='records', lines=False, force_ascii=False, indent=4)
        with open(self.filepath, 'w', encoding='utf-8') as file:
            file.write(json_str)

        for index, row in new_df.iterrows():
            key = (row['content'])
            self.cache[key] = True

if __name__ == '__main__':
    json_flie = 'contents_slice.json'
    cache_path = 'qa_slice_cache_inverse.json'

    df = pd.read_json(os.path.join(os.path.dirname(__file__), json_flie))
    print("Original columns:", df.columns)
    df = df.drop('title', axis=1)
    print("Updated columns:", df.columns)

    prompt = PromptTemplate.from_template(QA_GENERATE_PROMPT_TMPL)
    chain = LLMChain(llm=get_llm_model(), prompt=prompt,verbose=False)

    # get content column
    content = df['content']
    total_quries = len(content)
    all_questions = []

    cacher = AutoCahce(cache_path)
    for index in range(20000):
        i = total_quries - index - 1
        query = content[i]

        if cacher.check_key(query):
            cached_df = cacher.get_value(query)
            for key in cached_df.columns:
                if key not in df:
                    df[key] = None
            df.iloc[i] = cached_df.iloc[0]
            print(f"Skip annotating the example {i} since it is cached.")
        else:
            input = {'context_str' : query,
                'num_questions_per_chunk' : 2}
            response = chain.invoke(input)
            result = str(response['text']).strip().split("\n")
            questions = [re.sub(r"^\d+[\).\s]", "", question).strip() for question in result]

            if 'questions' not in df.columns:
                df['questions'] = pd.NaT
                df['questions'] = df['questions'].astype('object')

            questions = [question for question in questions if len(question) > 0]
            all_questions.append(questions)
            my_dict = dict(questions=questions)

            df.loc[i, my_dict.keys()] = my_dict.values()

            cacher.save_response(df.iloc[[i]])
            print(f"Generating {i}.")

    print("Synthetic dataset generation completed!")
