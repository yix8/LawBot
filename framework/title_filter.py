import json
from utils import *

def title_filter(load_path, saving_path = 'title_category.json'):
    with open(load_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    titles = set() 

    for item in data:
        if 'title' in item: 
            titles.add(item['title'])  
    print("Number of unique titles:", len(titles))

    titles_list = list(titles)
    with open(saving_path, 'w', encoding='utf-8') as file:
        json.dump(titles_list, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    title_filter('contents_repeat_index.json')
