import json
import os

file_path = os.path.join(os.path.dirname(__file__),'qa_full.json')

subset_file_path = os.path.join(os.path.dirname(__file__),'contents_slice_sub.json')

with open(subset_file_path, 'r', encoding='utf-8') as file:
    subset_data = json.load(file)

with open(file_path, 'r', encoding='utf-8') as file:
    full_data = json.load(file)

matching_elements = []

i = 0
for item in subset_data:
    value1 = item.get('content')  
    if value1 is not None:
        print(f"Find {i+1}.")
        for entry in full_data:
            if entry.get('content') == value1:
                matching_elements.append(entry)
    i += 1

print(f"Subset contains {len(matching_elements)} data.")
new_file_path = os.path.join(os.path.dirname(__file__), 'qa_subset.json')
with open(new_file_path, 'w', encoding='utf-8') as new_file:
    json.dump(matching_elements, new_file, ensure_ascii=False, indent=4)

print("Successfullly saved to:", new_file_path)