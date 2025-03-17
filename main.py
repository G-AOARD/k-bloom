import os
import json
import fire
import sys

from models.knowledge_extraction import KnowledgeExtraction
import subprocess as sp
import os
from threading import  Timer
import os
import torch.multiprocessing as mp
import time


def main(rel_set='conceptnet_new_T5',
         model_name='roberta-large',
         max_n_ent_tuples=1000,
         max_n_prompts=20,
         prompt_temp=2.,
         max_word_repeat=5,
         max_ent_subwords=2,
         top_k = 10,
         use_init_prompts=True,
         use_top1_prompt=False):
    old_stdout = sys.stdout

    # rel_set = 'bio_data_test'
    # model_name='bio-clinical-bert'
    knowledge_extraction = KnowledgeExtraction(
        model_name=model_name,
        max_n_ent_tuples=max_n_ent_tuples,
        max_n_prompts=max_n_prompts,
        max_word_repeat=max_word_repeat,
        max_ent_subwords=max_ent_subwords,
        prompt_temp=prompt_temp)

    relation_info = json.load(open(f'relation_info/{rel_set}.json'))

    print('------------------------ START EXTRACT RELATION --------------------------', flush=True)

    for rel, info in relation_info.items():
        extract_new_tuples_in_relation(rel,info, rel_set, model_name, knowledge_extraction, top_k, use_init_prompts, use_top1_prompt)

    print('------------------------ END EXTRACT RELATION --------------------------', flush=True)

    sys.stdout = old_stdout


def extract_new_tuples_in_relation(rel, info, 
                                    rel_set,
                                    model_name,
                                    knowledge_harvester,
                                    top_k,
                                    use_init_prompts = False,
                                    use_top1_prompt  = False):
    
    print(f'-----Extracting for relation {rel}...')

    setting = f'{knowledge_harvester.max_n_ent_tuples}tuples'
    if use_init_prompts:
        setting += '_initprompts'
    else:
        if use_top1_prompt == True:
            setting += f'_top1prompts'

        else:
            setting += f'_top{knowledge_harvester.max_n_prompts}prompts'

    output_dir = f'results/{rel_set}/{setting}/{model_name}'

    if os.path.exists(f'{output_dir}/{rel}/ent_tuples.json'):
        print(f'------- file {output_dir}/{rel}/ent_tuples.json exists, skipped.')
        return

    os.makedirs(f'{output_dir}/{rel}', exist_ok=True)
    json.dump([], open(f'{output_dir}/{rel}/ent_tuples.json', 'w'))

    knowledge_harvester.clear()

    knowledge_harvester.set_seed_ent_tuples(
        seed_ent_tuples=info['seed_ent_tuples'])

    if use_init_prompts == False:
        if use_top1_prompt == True:
            knowledge_harvester.set_prompts([info['prompts'][0]])
        else:
            knowledge_harvester.set_prompts(
                prompts=info['init_prompts'] if use_init_prompts
                else list(set(info['init_prompts'] + info['prompts'])))
    else:
        knowledge_harvester.set_prompts(
            prompts=info['init_prompts'])

    
    knowledge_harvester.set_initial_prompt(info['init_prompts'])
    # NEW APPROACH
    if os.path.exists(f'{output_dir}/{rel}/prompts.json'):
        print(f'------- file {output_dir}/{rel}/prompts.json exists, skipped.')
    else:        
        knowledge_harvester.update_prompts_new_approach(info['init_prompts'][0])
        json.dump([], open(f'{output_dir}/{rel}/prompts.json', 'w'))

        with open(f'{output_dir}/{rel}/prompts.json', 'w') as f: 
            json.dump(knowledge_harvester.weighted_prompts, f, indent=4) 

    # Update weight for new seed tuples based on initial prompt and created prompts
    if os.path.exists(f'{output_dir}/{rel}/raw_ent_tuples.json'):
        json.dump([], open(f'{output_dir}/{rel}/ent_tuples.json', 'w'))
        raw_tuples = json.load(open(f'{output_dir}/{rel}/raw_ent_tuples.json'))
        knowledge_harvester.set_weighted_ent_tuples(raw_tuples)

    else:
        print(f'------- Update weight for new prompts for rel: {rel}')
        json.dump([], open(f'{output_dir}/{rel}/ent_tuples.json', 'w'))

        if "roberta" in model_name:
            knowledge_harvester.update_ent_tuples(top_k, info['init_prompts'][0], False)
        else:
            knowledge_harvester.update_ent_tuples(top_k, info['init_prompts'][0], True)
        # Write the list to a JSON file
        with open(f'{output_dir}/{rel}/raw_ent_tuples.json', 'w') as json_file:
            json.dump(knowledge_harvester.weighted_ent_tuples, json_file, indent=4)   
        
    knowledge_harvester.select_ent_tuples()

    # Write the list to a JSON file
    with open(f'{output_dir}/{rel}/ent_tuples.json', 'w') as json_file:
        json.dump(knowledge_harvester.weighted_ent_tuples, json_file, indent=4)   

if __name__ == '__main__':
    fire.Fire(main)
