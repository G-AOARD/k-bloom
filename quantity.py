import collections
import json
import os
import sys
import fire


def main(folder_name , bioinfer):
    entities = list()
    entities = get_diversity_entities_in_folder(folder_name, entities)
    print("total entities: " , len(entities))    

def get_diversity_entities_in_folder(folder_name, tuples):
    for x in os.listdir(folder_name):
        if "ent_tuples" in x:
            # Prints only text file present in My Folder
            tuples = get_diversity_entities_in_file(folder_name + "/" + x, tuples)
        elif ".json" not in x and ".txt" not in x:
            tuples = get_diversity_entities_in_folder(folder_name + "" + x, tuples)
    print("total tuples: ", len(tuples) ," in folder: ", folder_name)
    return tuples


def get_diversity_entities_in_file(name_file, entities):
    with open(name_file) as f:
        data = f.read()
    # reconstructing the data as a dictionary
    js = json.loads(data)
    for i in js:
        entity_pair_head = i[0][0]
        entity_pair_tail = i[0][1]
        entities.append(entity_pair_head)
        entities.append(entity_pair_tail)

    print("total entities: ", len(entities) ," in file: ", name_file)
    return entities


if __name__ == '__main__':
    main("results/conceptnet/1000tuples_top1prompts/bert-base-cased/", True)
