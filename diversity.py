import json
import os
import sys
import fire


def main(folder_name, diversity):
    diversity = list()
    tuples = list()
    get_diversity_entities_in_folder(folder_name, diversity, tuples)

def get_diversity_entities_in_folder(folder_name, diversity, tuples):
    for x in os.listdir(folder_name):
        if "ent_tuples" in x and "raw_ent_tuples" not in x and "ent_tuples_final" not in x :
            # Prints only text file present in My Folder
            diversity, tuples = get_diversity_entities_in_file(folder_name + "/" + x, diversity, tuples)
        elif ".json" not in x and ".txt" not in x:
            diversity, tuples = get_diversity_entities_in_folder(folder_name + "" + x, diversity, tuples)
    print("Diversity of entites: ", len(diversity), " total tuples: ", len(tuples) , ' with diversity%: ', float(len(diversity)) /len(tuples)  ," in folder: ", folder_name)
    return diversity, tuples


def get_diversity_entities_in_file(name_file, diversity, tuples):
    with open(name_file) as f:
        data = f.read()

    # reconstructing the data as a dictionary
    js = json.loads(data)
    for i in js:
        entity_pair_head = i[0][0]
        entity_pair_tail = i[0][1]
        # tuples.append(entity_pair_head + "," + entity_pair_tail)
        tuples.append(entity_pair_head)
        tuples.append(entity_pair_tail)
        if entity_pair_head not in diversity:
            diversity.append(entity_pair_head)
        if entity_pair_tail not in diversity:
            diversity.append(entity_pair_tail)

    print("Diversity of entites: ", len(diversity), " total tuples: ", len(tuples) , " in file: ", name_file)
    return diversity, tuples

if __name__ == '__main__':
    diversity = list()
    main("results/conceptnet_new_T5/1000tuples_initprompts/roberta-large/", diversity)
    # main("results/conceptnet/1000tuples_top20prompts_original_result/bert-base-cased/", diversity)