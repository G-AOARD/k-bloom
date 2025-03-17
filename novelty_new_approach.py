import collections
import json
import os
import sys
import fire
from tqdm import tqdm
import Constants

def get_diversity_lama(folder_name):
    
    diversity = list()
    entities = list()
    tuples = list()
    diversity, entities, tuples = get_diversity_entities_in_folder(folder_name, diversity, entities, tuples)
    
    

def main(folder_name , original_str, is_conceptNet = True):
    if is_conceptNet == False:
        get_diversity_lama(folder_name)
        return
    diversity = list()
    entities = list()
    tuples = list()
    diversity, entities, tuples = get_diversity_entities_in_folder(folder_name, diversity, entities, tuples)

    with open(original_str) as f:
        data = f.read()

    original_ents = list()

    print("-----------------")
    # reconstructing the data as a dictionary
    with open(original_str, 'r') as fp, open("results/test.txt", 'w') as fp1:

        # read all lines using readline()
        lines = fp.readlines()
        for row in tqdm(lines):
            row = row.split("\n")[0]
            sub_str = row.split("\t")
            head_ent = sub_str[1]
            tail_ent = sub_str[2]
            if head_ent not in original_ents:
                fp1.write(head_ent)
                original_ents.append(head_ent)
            if tail_ent not in original_ents:
                fp1.write(tail_ent)
                original_ents.append(tail_ent)
    
    res = get_proportion_unique_entities(diversity, original_ents)

def get_proportion_unique_entities_1(folder_name , original_str):
    diversity = list()
    entities = list()
    tuples = list()
    diversity, entities, tuples = get_diversity_entities_in_folder(folder_name, diversity, entities, tuples)

    diversity1 = list()
    original_ents = list()
    tuples1 = list()
    diversity1, original_ents, tuples1 = get_all_entities_in_file(original_str, diversity1, original_ents, tuples1)
    
    res = get_proportion_unique_entities(diversity, diversity1)


def get_proportion_unique_entities(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    print('novelty new approach - 0: ', 1.0 - (float(len(lst3)/ len(lst1))))

    result = collections.Counter(lst1) & collections.Counter(lst2)

    intersected_list = list(result.elements())
    print('novelty new approach - 1: ', (float(len(lst1) - len(intersected_list)) / len(lst1)))
    return lst3

def get_diversity_entities_in_folder(folder_name, diversity, entities, tuples):
    for x in os.listdir(folder_name):
        if "summary" not in x and "accuracy_new_approach" not in x and "raw_ent_tuples" not in x and "acc_entities" not in x and "ent_tuples_final" not in x:
            if "ent_tuples" in x:
                # Prints only text file present in My Folder
                diversity, entities, tuples = get_diversity_entities_in_file(folder_name + "/" + x, diversity, entities, tuples)
            elif ".json" not in x and ".txt" not in x:
                diversity, entities, tuples = get_diversity_entities_in_folder(folder_name + "" + x, diversity, entities, tuples)
    print("Diversity of entites: ", len(diversity), " total entities: ", len(entities) , " total tuples: ", len(tuples)," in folder: ", folder_name)
    return diversity, entities, tuples


def get_diversity_entities_in_file(name_file, diversity, entities, tuples):
    with open(name_file) as f:
        data = f.read()

    # reconstructing the data as a dictionary
    js = json.loads(data)
    if len(js) > Constants.NUMBER_OF_ALL_TUPLES:
        js = js[:Constants.NUMBER_OF_ALL_TUPLES]
        with open(f'{name_file}', 'w') as f: 
            json.dump(js, f, indent=4) 
    
    for i in js:
        entity_pair_head = i[0][0]
        entity_pair_tail = i[0][1]
        entities.append(entity_pair_head)
        entities.append(entity_pair_tail)
        tuples.append([entity_pair_head, entity_pair_tail])
        if entity_pair_head not in diversity:
            diversity.append(entity_pair_head)
        if entity_pair_tail not in diversity:
            diversity.append(entity_pair_tail)

    print("Diversity of entites: ", len(diversity), " total entities: ", len(entities) , " total tuples: ", len(tuples)," in file: ", name_file)
    return diversity, entities, tuples

def get_all_entities_in_file(name_file, diversity, entities, tuples):
    with open(name_file) as f:
        data = f.read()

    # reconstructing the data as a dictionary
    js = json.loads(data)
    for i in js:
        entity_pair_head = i[0]
        entity_pair_tail = i[1]
        entities.append(entity_pair_head)
        entities.append(entity_pair_tail)
        tuples.append([entity_pair_head, entity_pair_tail])
        if entity_pair_head not in diversity:
            diversity.append(entity_pair_head)
        if entity_pair_tail not in diversity:
            diversity.append(entity_pair_tail)

    print("Diversity of entites: ", len(diversity), " total entities: ", len(entities) , " total tuples: ", len(tuples)," in file: ", name_file)
    return diversity, entities, tuples

if __name__ == '__main__':
    original_str = "results/conceptnet_new_T5/train300k_conceptnet.txt" 
    main("results/conceptnet_new_T5/10000tuples_top20prompts/roberta-large/", original_str, is_conceptNet=True)
