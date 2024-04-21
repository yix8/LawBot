import json
import os

script_dir = os.path.dirname(__file__)
input_file_path = os.path.join(script_dir, 'qa_subset.json')

with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)


output_data = []

for item in data:
    if 'questions' in item and isinstance(item['questions'], list) and item['questions']:
        if len(item['content']) <= 512:
            output_data.append(item)

new_file_path = os.path.join(os.path.dirname(__file__), 'qa_subset_filter.json')
with open(new_file_path, 'w', encoding='utf-8') as new_file:
    json.dump(output_data, new_file, ensure_ascii=False, indent=4)

print("Successfullly saved to:", new_file_path)