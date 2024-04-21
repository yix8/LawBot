import json
import os
import random

random.seed(20)

script_dir = os.path.dirname(__file__) 
input_file_path = os.path.join(script_dir, 'qa_subset_filter.json')  

with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  

output_data = []
for item in data:
    new_item = {'text': item['content']}
    output_data.append(new_item)

output_file_path = os.path.join(script_dir, 'pretrain.jsonl')


with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in output_data:
        json.dump(line, outfile, ensure_ascii=False) 
        outfile.write('\n')  

print("Pretraining Data has been processed and saved in JSONL format.")

finetune_data = []
all_indices = list(range(len(data)))

for i, item in enumerate(data):
    new_item = {
        'query': item['content'],  
        'pos': item['questions'][:2] 
    }

    available_indices = [idx for idx in all_indices if idx != i] 
    random_indices = random.sample(available_indices, 5)  

    neg_questions = []
    for idx in random_indices:
        question_list = data[idx]['questions']
        assert question_list is not None, f"{idx} has empty quetions."
        question = question_list[random.randint(0, len(question_list) - 1)]
        neg_questions.append(question)

    new_item['neg'] = neg_questions 

    finetune_data.append(new_item)

finetune_output_file_path = os.path.join(script_dir, 'finetune.jsonl')
with open(finetune_output_file_path, 'w', encoding='utf-8') as outfile:
    for line in finetune_data:
        json.dump(line, outfile, ensure_ascii=False) 
        outfile.write('\n') 

print("Finetune Data has been processed and saved in JSONL format.")



