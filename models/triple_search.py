import string
import torch
import heapq

from tqdm import tqdm

from data_utils_folder.data_utils import get_n_ents, get_mask_place, get_masked_prompt, get_n_masks ,stopwords

import numpy as np

import math

import Constants
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
import numpy as np
from bert_score import score
from models.scoring_triples import get_bm25_dict,word_mover_score_ours
from collections import defaultdict

import textstat
from typing import List, Union, Iterable

from itertools import zip_longest
from itertools import chain

from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from math import sqrt, pow, exp

class TripleSearch:
    def __init__(self, model):
        self._model = model

    def check_predicted_entites_in_heap(self, pred, collected_tuples_heap):
        for i, val in enumerate(collected_tuples_heap):
            head_ent = val[0]
            tail_ent = val[1]
            if pred[0] == head_ent or pred[1] == tail_ent:
                return True
        return False

    ###### Our Aprroach
    def search(self, raw_initial_prompt , weighted_prompts, seed_ent_tuples, max_word_repeat, max_ent_subwords, n, top_k, is_bert_model = True):
        seed_ent_tuples_sim = self._model.get_mean_embedding_seed_ent_tuples(seed_ent_tuples)

        # raw_tuples_1_1, raw_tuples_1_2, raw_tuples_2_1, raw_tuples_2_2  = self.search_entity(raw_initial_prompt, weighted_prompts, seed_ent_tuples_sim, top_k)  
        raw_tuples_1_1, raw_tuples_1_2, raw_tuples_2_1, raw_tuples_2_2  = self.search_entity_new_approach(raw_initial_prompt, weighted_prompts, seed_ent_tuples_sim, seed_ent_tuples,top_k, is_bert_model)
        all_ent_tuples = raw_tuples_1_1 + raw_tuples_1_2 + raw_tuples_2_1 + raw_tuples_2_2
        return all_ent_tuples

    def find_appropriate_tuples(self, all_ent_tuples, raw_initial_prompt, seed_ent_tuples):
        print("[entity_tuple_searcher] - Scoring all entities pairs")
        all_ent_tuples = self.kNearestNeighbor(all_ent_tuples, raw_initial_prompt, seed_ent_tuples)
        print("[entity_tuple_searcher] - End Scoring all entities pairs")
        return all_ent_tuples

    def kNearestNeighbor(self, ent_pairs, raw_initial_prompt, seed_ent_tuples):
        all_ent_tuples = []
        for i in tqdm(range(len(ent_pairs))):
            temp = dict()
            score = self.score_objective_value_ent_pair(ent_pairs[i], raw_initial_prompt, seed_ent_tuples)
            temp["ent_pair"] = ent_pairs[i][1]
            temp["sim_score"] = score
            print("----- score of ent pair: ", temp)
            all_ent_tuples.append(temp)

        top_k=len(all_ent_tuples) if(len(all_ent_tuples)) < Constants.NUMBER_OF_ALL_TUPLES else  Constants.NUMBER_OF_ALL_TUPLES
        # Sort the data based on the comparison function
        sorted_data = sorted(all_ent_tuples, key=self.compare)
        # Pick the top k distinguishable elements
        top_k_elements = []
        head_ent_dict = dict()
        tail_ent_dict = dict()
        for item in sorted_data:
            ent_pairs = item["ent_pair"]
            head_ent_condition, head_ent_dict  = self.check_key_in_dict(head_ent_dict, ent_pairs[0])
            tail_ent_condition, tail_ent_dict  = self.check_key_in_dict(tail_ent_dict, ent_pairs[1])

            if head_ent_condition == True or tail_ent_condition == True and len(top_k_elements) <= top_k:
                top_k_elements.append([item["ent_pair"], item["sim_score"]])
            
        return top_k_elements

    def check_key_in_dict(self, dictionary, target_key):
        if target_key in dictionary:
            if dictionary[target_key] <= Constants.FREQUENCY_KEY_AFTER_SORT:
                dictionary[target_key] += 1
                return True, dictionary
            else:
                return False, dictionary
        else:
            dictionary[target_key] = 1
            return True, dictionary

    def pick_top_k(self, ent_tuples, top_k):
        
        # Sort the data based on the comparison function
        sorted_data = sorted(ent_tuples, key=self.compare, reverse=True)
        # Pick the top k distinguishable elements
        top_k_elements = []
        head_ent_dict = dict()
        tail_ent_dict = dict()
        for item in sorted_data:
            ent_pairs = item["ent_pair"]
            head_ent_condition, head_ent_dict  = self.check_key_in_dict(head_ent_dict, ent_pairs[0])
            tail_ent_condition, tail_ent_dict  = self.check_key_in_dict(tail_ent_dict, ent_pairs[1])

            if head_ent_condition == True or tail_ent_condition == True and len(top_k_elements) <= top_k:
                top_k_elements.append([item["ent_pair"], item["sim_score"]])
            
        return top_k_elements

    def score_semantic_sim_ent_pair_with_all_prompts(self, ent_tuple, raw_initial_prompt, seed_ent_tuples):
        total_score  = 0.0
        paraphrased_sentence = raw_initial_prompt.replace("<ENT0>", ent_tuple[0])
        paraphrased_sentence = paraphrased_sentence.replace("<ENT1>", ent_tuple[1])

        for i in range(len(seed_ent_tuples)):
            seed_ent_tuple = seed_ent_tuples[i]
            for j in range(len(seed_ent_tuple)):
                seed_ent_tuple[j] = seed_ent_tuple[j].replace("_", " ")

            initial_sentence = raw_initial_prompt.replace("<ENT0>", seed_ent_tuple[0])
            initial_sentence = initial_sentence.replace("<ENT1>", seed_ent_tuple[1])
        
            bm25_dict_hyp = get_bm25_dict(paraphrased_sentence) 
            bm25_dict_ref = get_bm25_dict([initial_sentence]) 
            
            paraphrased_sentence = [paraphrased_sentence] * len([initial_sentence])
            emd_scores = word_mover_score_ours([initial_sentence], paraphrased_sentence, bm25_dict_ref, bm25_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
            sentence_score = np.mean(emd_scores)
            total_score += sentence_score
            
        total_score = total_score / len(seed_ent_tuples)
        return total_score
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def score_objective_value_ent_pair(self, gen_ent_pair, raw_initial_prompt, seed_ent_tuples):
        semantic_score = self.score_semantic_sim_ent_pair_with_all_prompts(gen_ent_pair[1], raw_initial_prompt, seed_ent_tuples)
        fluency_score = self.score_fluency_ent_pair_with_raw_initial_prompt(gen_ent_pair[1], raw_initial_prompt)
        h_semantic = self.sigmoid(semantic_score)
        h_fluency = self.sigmoid(fluency_score)
        total_score = gen_ent_pair[0] * (h_semantic * semantic_score + h_fluency * fluency_score)
        return total_score  

    def score_fluency_ent_pair_with_raw_initial_prompt(self, ent_tuple, raw_initial_prompt):
        score = 0.
        sen = raw_initial_prompt.replace("<ENT0>", ent_tuple[0])
        sen = sen.replace("<ENT1>", ent_tuple[1])
        score = self._model.score_fluency_of_sentence(sen)
        return score

    def search_entity_new_approach(self, raw_initial_prompt, weighted_prompts, seed_ent_tuples_sim, seed_ent_tuples, top_k, is_bert_model = True):
        print("[entity_tuple_searcher] Start searching all entity pairs")
        
        collected_tuples_heap_1_1 = []
        collected_tuples_heap_1_2 = []
        collected_tuples_heap_2_1 = []
        collected_tuples_heap_2_2 = []

        print("[entity_tuple_searcher] Start searching entity pairs 2-2")
        collected_tuples_heap_2_2 = self._model.get_ent_pairs_2_2(weighted_prompts, collected_tuples_heap_2_2, is_bert_model)

        print("[entity_tuple_searcher] Start searching entity pairs 2-1")
        collected_tuples_heap_2_1 = self._model.get_ent_pairs_2_1(weighted_prompts, collected_tuples_heap_2_1, is_bert_model)

        print("[entity_tuple_searcher] Start searching entity pairs 1-2")
        collected_tuples_heap_1_2 = self._model.get_ent_pairs_1_2(weighted_prompts, collected_tuples_heap_1_2, is_bert_model)

        print("[entity_tuple_searcher] Start searching entity pairs 1-1")
        collected_tuples_heap_1_1 = self._model.get_ent_pairs_1_1(weighted_prompts, collected_tuples_heap_1_1, is_bert_model)


        print("[entity_tuple_searcher] End searching all entity pairs")

        return collected_tuples_heap_1_1, collected_tuples_heap_1_2, collected_tuples_heap_2_1, collected_tuples_heap_2_2

    
    def find_positions(self, text, substring, token=" "):
        tokens = text.split(token)
        start = 0
        for token1 in tokens:
            if token1 == substring:
                return start
            start+=1
        return -1
    