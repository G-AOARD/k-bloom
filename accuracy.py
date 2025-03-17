import collections
import json
import os
def main(folder_name):
    diversity = list()
    tuples = list()
    diversity, tuples = get_diversity_entities_in_folder(folder_name, diversity, tuples)

    original_str = "results/bioinfer/bioinfer_test_all_entities.json"

    with open(original_str) as f:
        data = f.read()

    original_ents = list()
    # reconstructing the data as a dictionary
    js = json.loads(data)
    for i in js.keys():
        for j in js[i]:
            for k, val in enumerate(js[i][j]):
                head_ent = val[0]
                tail_ent = val[1]
                original_ents.append(head_ent)
                original_ents.append(tail_ent)

    get_accuracy_entities(diversity, original_ents)

def get_accuracy_entities(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    print('acc: ',(float(len(lst3)/ len(lst2))))

    result = collections.Counter(lst1) & collections.Counter(lst2)

    intersected_list = list(result.elements())
    print('acc: ', (float(len(intersected_list)) / len(lst2)))
    return lst3

def get_diversity_entities_in_folder(folder_name, diversity, tuples):
    for x in os.listdir(folder_name):
        if "ent_tuples" in x:
            # Prints only text file present in My Folder
            diversity, tuples = get_diversity_entities_in_file(folder_name + "/" + x, diversity, tuples)
        elif ".json" not in x and ".txt" not in x:
            diversity, tuples = get_diversity_entities_in_folder(folder_name + "" + x, diversity, tuples)
    print("Diversity of entites: ", len(diversity), " total tuples: ", len(tuples) ," in folder: ", folder_name)
    return diversity, tuples


def get_diversity_entities_in_file(name_file, diversity, tuples):
    with open(name_file) as f:
        data = f.read()

    # reconstructing the data as a dictionary
    js = json.loads(data)
    for i in js:
        entity_pair_head = i[0][0]
        entity_pair_tail = i[0][1]
        tuples.append(entity_pair_tail)
        tuples.append(entity_pair_head)
        if entity_pair_head not in diversity:
            diversity.append(entity_pair_head)
        if entity_pair_tail not in diversity:
            diversity.append(entity_pair_tail)

    print("Diversity of entites: ", len(diversity), " total tuples: ", len(tuples) ," in file: ", name_file)
    return diversity, tuples

if __name__ == '__main__':
    diversity = list()
    tuples = list()
    main("results/bioinfer/1000tuples_top20prompts/bio-bert/")