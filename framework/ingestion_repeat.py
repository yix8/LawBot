from utils import *

import os
import json

import re
from glob import glob

def extract_laws(text):
    pattern = r"(第[\d零一二三四五六七八九十百千]+条\s.*?)(?=第[\d零一二三四五六七八九十百千]+条\s|$)"
    laws = re.findall(pattern, text, re.DOTALL)
    return laws

def laws_count(content):
    pattern = r"^第([零一二三四五六七八九十百千万]+)条\s"
    all_laws = re.findall(pattern, content, re.MULTILINE)
    return len(all_laws)

def pre_process_files(content):
    main_title_pattern = r"^# [^\#].*$"
    main_title = re.findall(main_title_pattern, content, re.MULTILINE)
    if main_title:
        main_title = main_title[0]
    else:
        main_title = "No main title found"

    content = re.sub(r"^#+ .*$", "", content, flags=re.MULTILINE)

    return main_title, content 

def repeat_title(text, title):
    pattern = r"^第([零一二三四五六七八九十百千万]+)条\s"
    match = re.search(pattern, text)
    assert match != None, f"Content Error"

    if match:
        matched_text = match.group(0)
        replacement = title + "-" + matched_text
        replacement = replacement * 3

        result_text = re.sub(pattern, replacement, text, count=1)
        return result_text, matched_text[:-1]

def doc2vec(saving_path = 'nodes.json'):
    dir_path = os.path.join(os.path.dirname(__file__), './vanilla_chinese_laws')
    documents = []
    count = 0
    empty = []
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)  
            
            if file_name.endswith('.md'):
                print(f'Loading {file_name}.')
                count += 1
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    main_title, updated_content = pre_process_files(content)
                    laws = extract_laws(updated_content)
                    if(laws_count(content) == 0):
                        empty.append(file_name)
                    assert laws_count(content) != 0, f"Expected 0 laws."
                    assert len(laws) == laws_count(content), f"Expected {laws_count(content)} laws, but extracted {len(laws)} laws."

                    for index, law in enumerate(laws, start=1):
                        filtered_law = law.replace('\n', '')
                        repeated_law, index = repeat_title(filtered_law, main_title[2:])
                        law_data = {
                            "title": main_title[2:],
                            "index": index,
                            "origin": main_title[2:] + filtered_law,
                            "content": repeated_law
                        }
                        documents.append(law_data)

    print(f"{count} files are loaded.")
    with open(saving_path, 'w', encoding='utf-8') as json_file:
        json.dump(documents, json_file, ensure_ascii=False, indent=4)

    print("JSON data has been saved.")

if __name__ == '__main__':
    doc2vec('contents_repeat_index.json')