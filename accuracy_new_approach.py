import collections
import json
import os
import csv  

import requests
from tqdm import tqdm

import requests

from bardapi import BardCookies
import random
import os
import time
header = ['subject', 'relation', 'object', 'checked','isvalid']

def get_all_initial_prompt(file_name):
    initial_prompts = []

    with open(file_name) as f:
        contents = f.readlines()
        for i in range(len(contents)):
            initial_prompts.append(contents[i])
    return initial_prompts

def main(folder_name, initial_prompts, conceptnet_dataset = True):
    
    get_all_ent_pairs(folder_name)
    diversity = list()
    tuples = list()
    # create_csv_for_all_entities_file(folder_name)
    ent_pairs = get_random_ent_pairs(folder_name, 4,initial_prompts, conceptnet_dataset)
    
    get_accuracy_ent_pairs(folder_name, ent_pairs)

    #total_tuples, total_accuracy = get_all_entity_pairs_in_folder(folder_name)

def create_csv_for_all_entities_file(folder_name):
    with open(folder_name + 'all_entities.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        for x in tqdm(os.listdir(folder_name)):
            if(os.path.isdir(folder_name + "/" + x)) == True:
                for y in os.listdir(folder_name + "/" + x):
                    if "ent_tuples" in y:
                        with open(folder_name + "/" + x + "/" + y) as f:
                            data = f.read()
                        # reconstructing the data as a dictionary
                        js = json.loads(data)
                        for i in js:
                            # write the data
                            writer.writerow([i[0][0], x, i[0][1], False,0])
                        
def random_and_pick_top_k_ent_pairs(entities_pairs,k):
    random.shuffle(entities_pairs)
    results = random.sample(entities_pairs, k)
    return results

def get_all_ent_pairs(folder_name):
    ent_pairs = []
    for x in tqdm(os.listdir(folder_name)):
        if(os.path.isdir(folder_name + "/" + x)) == True:
            for y in os.listdir(folder_name + "/" + x):
                if "ent_tuples" in y and "raw_ent_tuples" not in y:
                    with open(folder_name + "/" + x + "/" + y) as f:
                        data = f.read()
                    # reconstructing the data as a dictionary
                    js = json.loads(data)
                    for i in js:
                        # write the data
                        ent_pairs.append([i[0][0], x, i[0][1]])
    print("total pairs: ", len(ent_pairs))
    return ent_pairs

from collections import Counter
def get_random_ent_pairs(folder_name, k, initial_prompts, conceptnet_dataset = True):
    ent_pairs = []
    idx = 0
    for x in tqdm(os.listdir(folder_name)):
        if(os.path.isdir(folder_name + "/" + x)) == True:
            for y in os.listdir(folder_name + "/" + x):
                if "ent_tuples" in y and "raw_ent_tuples" not in y:
                    with open(folder_name + "/" + x + "/" + y) as f:
                        data = f.read()
                    # reconstructing the data as a dictionary
                    js = json.loads(data)
                    temp_pairs = []
                    initial_prompt = ""
                    for j in range(len(initial_prompts)):
                        if conceptnet_dataset == True:
                            rel_name = initial_prompts[j].split("_")[0]
                        else:
                            rel_name = initial_prompts[j].split(":")[0]

                        if rel_name == x:
                            if conceptnet_dataset == True:
                                initial_prompt = initial_prompts[j].split("_")[1]
                            else:
                                initial_prompt = initial_prompts[j].split(":")[1]
                            break
                    tuples = dict()
                    ents = dict()
                    for i in js:
                        # write the data
                        if i[0][0] in ents:
                            # Key exists, increase the value
                            ents[i[0][0]] += 1
                        else:
                            # Key doesn't exist, add a new key-value pair
                            ents[i[0][0]] = 1

                        # write the data
                        if i[0][1] in ents:
                            # Key exists, increase the value
                            ents[i[0][1]] += 1
                        else:
                            # Key doesn't exist, add a new key-value pair
                            ents[i[0][1]] = 1

                        key = (i[0][0], x, i[0][1], initial_prompt)
                        if key in tuples:
                        # Key exists, increase the value
                            tuples[key] += 1
                        else:
                        # Key doesn't exist, add a new key-value pair
                            tuples[key] = 1

                        temp_pairs.append([i[0][0], x, i[0][1], initial_prompt])

                    temp_pairs = random_and_pick_top_k_ent_pairs(temp_pairs, k)
                    ent_pairs = ent_pairs + temp_pairs
                    # Calculate the frequency of each element in the dictionary
                    element_frequency = Counter(tuples.values())

                    # Filter elements with frequency higher than 10
                    elements_higher_than_1 = {key: value for key, value in tuples.items() if value > 1}

                    # Calculate the frequency of each element in the dictionary
                    element_frequency = Counter(ents.values())

                    # Filter elements with frequency higher than 10
                    elements_higher_than_10 = {key: value for key, value in ents.items() if value > 10}
                    if len(elements_higher_than_1) > 0:
                         print("elements_higher_than_1 - RELATION: ", x)
                    if len(elements_higher_than_10) > 0:
                         print("elements_higher_than_10 - RELATION: ", x)
                    #print("elements_higher_than_1: ", elements_higher_than_1, " elements_higher_than_10: ", elements_higher_than_10)
    return ent_pairs

def get_accuracy_ent_pairs(folder_name, ent_pairs):

    acc = 0
    for i in tqdm(range(len(ent_pairs))):
        text = ent_pairs[i][3].replace("<ENT0>", ent_pairs[i][0])
        text = text.replace("<ENT1>", ent_pairs[i][2])

        is_valid = search_tuple_google(text)
        if is_valid == True:
            print('tuple',ent_pairs[i] , " valid")
            acc +=1
        else:
            print('tuple',ent_pairs[i] , " INVALID")

        
        print("--- element: ", i, " - accuracy: ", acc)

    return acc

#####

cookie_dict = {
        "__Secure-1PSID": "g.a000ggjZaa442sEEZo6t1wVYSPvshdvWfzLjF0dHp4AshnO_b32vaUgoSGZv4KuY0jJZm_8jagACgYKAW0SAQASFQHGX2MiCtHRcidGZq8MwHTKGOGeTxoVAUF8yKoPh6Nei_srmqvsAqizhhb90076",
        "__Secure-1PSIDTS": "sidts-CjIBYfD7Z7GxjyZCzS0t25xb10K8TK8y6bZRaEsm9dhRE5F2z5e3bSq4zQWKFPVylGnUNRAA",
        "__Secure-1PSIDCC": "ABTWhQHYIX82LBQV6tpsvR6Mk9Bd0w3h2hVeKyiCWtbWAQeMTMWVdV3zHZy7sCQ9jl0H6HyrNzvF",
    }
def search_tuple_google(user_query):

    bard = BardCookies(cookie_dict=cookie_dict)
    bard.max_redirects = 80
    response = bard.get_answer("is the statement correct or not: " + user_query)['content']

    if "incorrect" in response:
        print("query: ", user_query, " object: ", object ," is NOT valid")
        return False
    
    print("query: ", user_query, " object: ", object ," is valid")
    return True
    

def check_url_contain_string(url, sub_string):

    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the content as text
            if sub_string in response.text:
                return True
            return False

        else:
            # Handle unsuccessful response (e.g., print an error message)
            print(f'Failed to fetch content. Status code: {response.status_code}')
            return False

    except requests.exceptions.RequestException as e:
        # Handle any exceptions that may occur during the request
        print(f'An error occurred: {e}')
    return False


def bio_bert(folder_name, initial_prompts):
    
    get_all_ent_pairs(folder_name)
    diversity = list()
    tuples = list()
    # create_csv_for_all_entities_file(folder_name)
    ent_pairs = get_random_ent_pairs(folder_name, 16,initial_prompts, False)
    
    get_accuracy_ent_pairs(folder_name, ent_pairs)

# print(f'{google_search("giraffes location", "africa")}')

def get_unchecked_bio_data_long_text():
    rows = []
    with open('bio_data_test_long_sent.csv', 'r', newline='') as file:
        csvreader = csv.reader(file)
        _ = next(csvreader)
        for row in csvreader:
            rows.append(row)
        
    idx = 0
    for row in rows:
        if row[3] == "" and idx < 70:
            entities = row[2].split(", ")
            text = row[1].replace("<ENT0>", entities[0])
            text = text.replace("<ENT1>", entities[1])

            is_valid = search_tuple_google(text)
            if is_valid == True:
                row[3] = "Valid"
                print('tuple',row , " valid")
            else:
                row[3] = "Not Valid"
                print('tuple',row , " INVALID")
            idx +=1
                    
    with open('bio_data_test_long_sent.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["relation", "initial prompt", "entity pair","result"]   
        writer.writerow(field)
        for row in rows:
            writer.writerow(row)
            
if __name__ == '__main__':
    diversity = list()
    tuples = list()
    initial_prompts = get_all_initial_prompt("results/bio_data_test_long_sent/initial_prompts.txt")
    main("results/lama_new_20_rel/1000tuples_top1prompts/roberta-large/", initial_prompts, conceptnet_dataset = False)


