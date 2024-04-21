import json
# import cohere
import random
import os
import agent

random.seed(20)

'''
# initialize the Cohere Client with an API Key
co = cohere.Client("Ahe3QHGu0Glc07UbOGAJmuIgTHd2ISCj9s8JJXF0")


def chat(message, documents=None):
    response = co.chat(
        model="command",
        message=message,
        documents=documents,
        prompt_truncation="AUTO",
    )
    #token = response.token_count["billed_tokens"]
    #print(token)
    return response.text
'''
SAMPLE_CACHE_PATH = "./predictions/cache/sample_cache.json"
DONE_PATH = "./predictions/cache/done.json"


def check_file(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)


if __name__ == "__main__":
    '''
    mean_len_list = [93.74199999999985, 287.40999999999997, 213.61400000000017, 208.27399999999966,
                     1006.0519999999989, 318.4379999999999, 301.21000000000015, 1367.0100000000004,
                     192.758, 880.627999999999, 772.3340000000002, 264.308, 786.9680000000001,
                     210.20599999999965, 691.1499999999997, 830.8159999999997, 1153.5840000000012,
                     383.43399999999957, 649.5840000000003, 243.61999999999998]
    '''
    mean_len_list = [93.74199999999985, 287.40999999999997, 210.20599999999965, 786.9680000000001]
    sample_num =100
    folder_path = "./evaluation_data/zero_shot"
    agent = agent.Agent()
    task = 0
    check_file(SAMPLE_CACHE_PATH)
    check_file(DONE_PATH)
    with open(SAMPLE_CACHE_PATH, 'r', encoding="utf-8") as file1:
        sample_cache = json.load(file1)
    with open(DONE_PATH, 'r', encoding="utf-8") as file2:
        done = json.load(file2)

    for file_name in ['1-1.json', '1-2.json', '3-2.json', '3-1.json']:
        file_path = os.path.join(folder_path, file_name)
        answer_dict_path = os.path.join("./predictions/zero_shot/rag", file_name)
        check_file(answer_dict_path)
        with open(answer_dict_path, "r", encoding="utf-8") as f:
            answer_dict = json.load(f)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            short_list = []
            for i in range(500):
                message = data[i]["instruction"] + "\n" + data[i]["question"]
                if len(message) < mean_len_list[task]:
                    short_list.append(i)
            if file_name not in sample_cache:
                sample_list = random.sample(short_list, sample_num)
                sample_cache[file_name] = sample_list
                with open(SAMPLE_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(sample_cache, f, ensure_ascii=False, indent=4)
            else:
                sample_list = sample_cache[file_name]
            for i in sample_list:
                if file_name not in done:
                    done[file_name] = []
                if i not in done[file_name]:
                    message = data[i]["instruction"] + "\n" + data[i]["question"]
                    response = agent.retrival_law_func('x', message)
                    pred = {"origin_prompt": [{
                        "role": "HUMAN",
                        "prompt": data[i]["instruction"] + "\n" + data[i]["question"]
                    }],
                    "prediction": response,
                    "refr": data[i]["answer"]}
                    answer_dict[str(i)] = pred
                    with open(answer_dict_path, "w", encoding="utf-8") as f:
                        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
                    done[file_name].append(i)
                    print(file_name, len(done[file_name]))
                    with open(DONE_PATH, "w", encoding="utf-8") as f:
                        json.dump(done, f, ensure_ascii=False, indent=4)
        task += 1



