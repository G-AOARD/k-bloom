import string
import torch
from copy import deepcopy
from enum import Enum
import math
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

from data_utils_folder.data_utils import get_n_ents, get_n_ents, \
get_sent, find_sublist ,stopwords, find_position_mask_in_prompt
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import operator

import re
import Constants
from heapq import nlargest

import sys
import torch.nn.functional as F

from tqdm import tqdm

import math
import heapq
from models.scoring_triples import get_idf_dict, collate_idf, collate_token_ids

from itertools import islice
# Load GloVe vectors
from sklearn.metrics.pairwise import (
    cosine_similarity
)
from sentence_transformers import SentenceTransformer
from rouge import Rouge

from nltk.translate.bleu_score import sentence_bleu

from huggingface_hub import notebook_login

import Constants
rouge_scorer = Rouge()


NUMBER_OF_INPUT_ENT_WITH_PROMPTS = 10 # create # of input sentences for extracting 1-2, 2-1, 2-2
TOP_K_PAIR = 3

class Tuples(Enum):
    Type_1_1 = 1
    Type_1_2 = 2
    Type_2_1 = 3
    Type_2_2 = 4

class LanguageModelWrapper:
    def __init__(self, model_name):
        model_name = "microsoft/deberta-v3-large"
        
        self._model_name = model_name
        if(model_name == 'bio-bert'):
            model_name = "dmis-lab/biobert-base-cased-v1.2"
        if model_name == "bio-clinical-bert":
            model_name = "emilyalsentzer/Bio_ClinicalBERT"

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self._tokenizer = AutoTokenizer.from_pretrained(model_name,  use_fast=False)

        if model_name == "emilyalsentzer/Bio_ClinicalBERT":
            self._model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


        self._model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states = True)

        self._model.eval()
        self._model.to(self._device)

        self._banned_ids = None
        self._get_banned_ids()

        self.sentence_transformer_model = SentenceTransformer("bert-base-nli-mean-tokens")


    def _get_banned_ids(self):
        self._banned_ids = self._tokenizer.all_special_ids
        for idx in range(self._tokenizer.vocab_size):
            if self._tokenizer.decode(idx).lower().strip() in stopwords:
                self._banned_ids.append(idx)

    def get_last_hidden_states_of_mask(self, input_text):
        with torch.no_grad():
            inputs = self.tokenizer(input_text, return_tensors="pt").to('cuda')
            tokens = self.tokenizer.tokenize(input_text)

            outputs = self.model(**inputs)

        # Retrieve the last hidden state
        last_hidden_state = outputs.last_hidden_state

        # Retrieve the embedding vector for the masked token
        masked_token_index = tokens.index("[MASK]")
        embedding_vector = last_hidden_state[0][masked_token_index]
        return outputs.hidden_states # contains the hidden representations for each token in each sequence of the batch
    
    def get_last_hidden_states(self, input_text):
        with torch.no_grad():
            inputs = self.tokenizer(input_text, return_tensors="pt").to('cuda')
            outputs = self.model(**inputs)
        
        hidden_states = [t.detach().cpu() for t in outputs.hidden_states]

        return outputs.hidden_states, hidden_states  # contains the hidden representations for each token in each sequence of the batch

    def convert_tuple_tensor_to_array(self, tuple_array):
        array = []
        for i, value in enumerate(tuple_array):
            array_item = value.detach().cpu()
            array.append(array_item)
        return array

    def get_mask_logits(self, input_text):
        with torch.no_grad():
            mask_input = self.tokenizer.encode(input_text, return_tensors="pt").to('cuda')
            mask_token_logits = self.model(mask_input)[0].squeeze().detach()
            is_masked = torch.where(mask_input == self.tokenizer.mask_token_id, 1, 0)
            masked_idxs = torch.nonzero(is_masked)
            probs= torch.softmax(mask_token_logits[masked_idxs[:,1]], dim=1)
        return probs

    def fill_ent_tuple_in_prompt(self, prompt, ent_tuple):

        assert get_n_ents(prompt) == len(ent_tuple)

        ent_tuple = deepcopy(ent_tuple)
        for ent_idx, ent in enumerate(ent_tuple):
            if prompt.startswith(f'<ENT{ent_idx}>'):
                ent_tuple[ent_idx] = ent.capitalize()
        sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)
        mask_spans = self.get_mask_spans(prompt=prompt, ent_tuple=ent_tuple)

        mask_positions = []
        for mask_span in mask_spans:
            mask_positions.extend([pos for pos in range(*mask_span)])

        masked_inputs = self.tokenizer(
            [sent] * len(mask_positions), return_tensors='pt').to('cuda')
        label_token_ids = []
        for i, pos in enumerate(mask_positions):
            label_token_ids.append(masked_inputs['input_ids'][i][pos].item())
            masked_inputs['input_ids'][i][mask_positions[i:]] = self.tokenizer.mask_token_id

        with torch.no_grad():
            logits = self.model(**masked_inputs).logits
            logprobs = torch.log_softmax(logits, dim=-1)

        mask_logits = []
        for i,_ in enumerate(mask_positions):
            mask_logits.append(logits[i][mask_positions[i]])

        mask_logprobs = logprobs[torch.arange(len(mask_positions)), mask_positions,label_token_ids].tolist()

        torch.cuda.empty_cache()

        return {
            'input_ids': self.tokenizer.encode(sent),
            'mask_spans': mask_spans,
            'mask_positions': mask_positions,
            'mask_logprobs': mask_logprobs,
            'mask_logits': mask_logits,
        }

    def get_mask_spans(self, prompt, ent_tuple):

        assert get_n_ents(prompt) == len(ent_tuple)
        print('[language_model_wrapper] get_mask_spans with prompt: ', prompt, ' and ent_tuple: ', ent_tuple)

        sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)
        input_ids = self._tokenizer.encode(sent)

        mask_spans = []
        for ent_idx, ent in enumerate(ent_tuple):
            prefix = prompt[:prompt.find(f'<ENT{ent_idx}>')].strip()
            for i in range(len(ent_tuple)):
                prefix = prefix.replace(f'<ENT{i}>', ent_tuple[i])
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)

            # processing -ing, -s, etc.
            ent_in_sent = prompt[prompt.find(f'<ENT{ent_idx}>'):].split()[0]
            for punc in string.punctuation:
                if punc not in '<>':
                    ent_in_sent = ent_in_sent.split(punc)[0]
            ent_in_sent = ent_in_sent.replace(f'<ENT{ent_idx}>', ent)

            # only mask the first word in an entity to
            # encourage entities with multiple words
            ent_in_sent = ent_in_sent.split()[0]

            ent_token_ids = self.tokenizer.encode(
                f' {ent_in_sent}' if sent[len(prefix)] == ' ' else ent_in_sent,
                add_special_tokens=False)

            if len(prefix_ids) > 0:
                l = find_sublist(input_ids, prefix_ids) + len(prefix_ids)
            else:
                l = find_sublist(input_ids, ent_token_ids)
            r = l + len(ent_token_ids)

            assert input_ids[l:r] == ent_token_ids
            mask_spans.append([l, r])
        print('[language_model_wrapper] get_mask_spans with mask_spans: ', mask_spans)
        return mask_spans
    
    ####### New Aprroach
    def find_indices(self, list_to_check, item_to_find):
        indices = []
        for idx, value in enumerate(list_to_check):
            if value == item_to_find:
                indices.append(idx)
        return indices

    def append_word_into_position(self, input_text, word_to_append, position):
        words = input_text.split()
        words.insert(position, word_to_append)
        return ' '.join(words)

    def find_positions(self,string, substring):
        positions = []
        start = 0
        while True:
            index = string.find(substring, start)
            if index == -1:
                break
            positions.append(index)
            start = index + 1
        return positions          
    
    def check_entities_valid(self, entities):
        error_counter =0
        for i in range(0, len(entities)):
            valid = self.check_entity_valid(entities[i])
            if valid == False:
                error_counter += 1
            if error_counter >= (len(entities)/2):
                return False
        return True

    def check_entity_valid(self, entity):
        regex = re.compile('[@_!#$%^&*()<>?/\|}{~:.]')

        # filter "the xxx", "new xxx", etc.
        if any([word in stopwords or word.lower() in stopwords for word in entity.split()]):
            return False

        if (regex.search(entity) != None):
            return False

        # filter entity with less than 3 characters
        if len(entity.replace(' ', '')) <= 2:
            return False

        # filter entity with single-character words
        if min([len(t) for t in entity.split()]) <= 1:
            return False

        # filter entity full of short words
        if max([len(t) for t in entity.split()]) <= 2:
            return False

        # filter entity with repeating words, e.g., "word word"
        if len(entity.split()) > 1 and len(set(entity.split())) == 1:
            return False
        
        return True
    
    def cosine_similarity(self,a, b):
        dot_product = a * b
        magnitude_a = abs(a)
        magnitude_b = abs(b)
        similarity = dot_product / (magnitude_a * magnitude_b)
        return similarity
        
    def take(self, n, iterable):
        """Return the first n items of the iterable as a list."""
        return list(islice(iterable, n))
    
    def avg_embedding_relation_similarity_candidate_pair(self, candidate_pair, seed_ent_pairs):
        total_score = 0.0
        for i in range(len(seed_ent_pairs)):
            seed_ent_pair = seed_ent_pairs[i]
            res = self.embedding_relation_similarity_two_pairs(candidate_pair=candidate_pair, seed_ent_pair=seed_ent_pair)
            total_score += res
        total_score /= len(seed_ent_pair)
        return total_score
            
    def embedding_relation_similarity_two_pairs(self, candidate_pair, seed_ent_pair):
        head_origin_embed, tail_origin_embed = self.get_mean_logits_two_words(seed_ent_pair[0], seed_ent_pair[1])
        head_cand_embed, tail_cand_embed = self.get_mean_logits_two_words(candidate_pair[0], candidate_pair[1])
        res = (tail_cand_embed - head_cand_embed) - (tail_origin_embed - head_origin_embed)
        return res

    # Function to calculate token-level perplexity
    def get_logits(self, sentence):
        tokens = self._tokenizer(sentence, return_tensors="pt").to(self._device)
        outputs = self._model(**tokens)
        logits = outputs.logits[0]
        probs = logits.softmax(dim=-1)
        #   pp = math.exp(- probs.mean().item())
        return probs

    def get_perplexity(self, prob):
        pp = math.exp(- prob)
        return pp

    def score_fluency_of_sentence(self, sentence):
        idf_dict = get_idf_dict(sentence)
        _, padded_idf, _, _, _ = collate_idf([sentence], self._tokenizer.tokenize, self._tokenizer.convert_tokens_to_ids,
                                                            idf_dict,device=self._device)

        tokens_ids = collate_token_ids([sentence], self._tokenizer.tokenize, self._tokenizer.convert_tokens_to_ids,
                                                            idf_dict,device=self._device)
        token_pps = []
        log_prob = self.get_logits(sentence)
        for i in range(1, len(tokens_ids[0])):
            values = self.get_perplexity(log_prob[i][tokens_ids[0][i-1]].item())
            token_pps.append(values)

        summed_log_prob = sum(token_pps)

        sum_idf = sum([padded_idf[0][i].item() for i in range(0, len(padded_idf[0]))])

        sentence_pp = math.exp(-1 * summed_log_prob / len(token_pps))
        fluency = (1.0 / sentence_pp) * sum_idf
        return fluency
    
    def get_mean_logits_two_words(self, word1, word2):
        word1 = word1.replace("_", " ")
        encoded_input1 = self._tokenizer(word1, return_tensors='pt').to(self._device)
        model_output1 = self._model(**encoded_input1)['logits'][0]
        token_embeddings1 = torch.mean(model_output1)

        word2 = word2.replace("_", " ")
        encoded_input2 = self._tokenizer(word2, return_tensors='pt').to(self._device)
        model_output2 = self._model(**encoded_input2)['logits'][0]
        token_embeddings2 = torch.mean(model_output2)

        return token_embeddings1.detach().cpu().numpy().item(), token_embeddings2.detach().cpu().numpy().item()
    
    def get_mean_embedding_seed_ent_tuples(self, seed_ent_tuples):
        result = dict()
        for ent_tuple in seed_ent_tuples:
            ent_tuple1 = [ent.replace('_', ' ') for ent in ent_tuple]
            token_embeddings1, token_embeddings2 = self.get_mean_logits_two_words(ent_tuple1[0], ent_tuple1[1])
            result[ent_tuple[0], ent_tuple[1]] = (token_embeddings1, token_embeddings2)
        return result

    def is_ent0_before_ent1_raw_prompt(self, raw_prompt):
        idx_ent0 = raw_prompt.find("<ENT0>")
        idx_ent1 = raw_prompt.find("<ENT1>")

        return idx_ent0 < idx_ent1 if idx_ent0 >= 0 and idx_ent1 >= 0 else False

    def get_ent0_1_1(self, weighted_prompts, collected_tuples_heap, is2_2 = False, is_bert_model = True):
        # Fint <ENT0>
        mask_logits_total = None

        for raw_prompt, weight in weighted_prompts:
            prompt = ""
            if is_bert_model == True:
                prompt = raw_prompt.replace(
                    '<ENT0>',"[MASK]")

                prompt = prompt.replace(
                    '<ENT1>',"[MASK]")
            else:
                prompt = raw_prompt.replace(
                    '<ENT0>',"<mask>")

                prompt = prompt.replace(
                    '<ENT1>',"<mask>")

            input_text = prompt
            
            mask_logits = self.get_mask_logits(input_text=input_text)

            mask_needed_index = 0
            if self.is_ent0_before_ent1_raw_prompt(raw_prompt) == False:
                mask_needed_index = 1
                if is2_2 == True:
                    mask_needed_index = 2
            
            mask_logits = mask_logits[mask_needed_index]

            if mask_logits_total is None:
                mask_logits_total = torch.zeros_like(mask_logits)
            mask_logits_total = mask_logits_total + mask_logits * weight

        mask_logits_total = mask_logits_total / sum(weight for _, weight in weighted_prompts)

        mask_logits_total[self.banned_ids] = -float('inf')
        logprobs = torch.log_softmax(mask_logits_total, dim=-1)
        logprobs, pred_ids = torch.sort(logprobs, descending=True)

        cur_logprobs = []
        collected_ent_heap = []
        cur_ent_tuple = []
        repeat_cnt = {}
        logprob_threshold = float('-inf') if len(collected_tuples_heap) < Constants.NUMBER_OF_ALL_TUPLES \
            else collected_tuples_heap[0][0]
        
        for logprob, pred_id in zip(logprobs, pred_ids):
            min_logprob_upd = min(cur_logprobs + [logprob.item()])
            if len(collected_ent_heap) == Constants.NUMBER_OF_ALL_TUPLES  and \
                    min_logprob_upd < collected_ent_heap[0][0]:
                break

            if min_logprob_upd < logprob_threshold:
                break

            if not any([ch.isalpha() for ch in
                        self.tokenizer.decode(pred_id)]):
                continue

            if any([punc in self.tokenizer.decode(pred_id)
                    for punc in string.punctuation]):
                continue

            cur_logprobs = cur_logprobs + [logprob.item()]
            pred_ent = self.tokenizer.decode(pred_id)

            pred_ent = pred_ent.strip().lower()
            if self.check_entity_valid(pred_ent) == False:
                continue
            for ent in cur_ent_tuple:
                # filter repeating entity in the entity tuple,
                # e.g., "grassland" vs "grass land"
                if pred_ent.replace(' ', '') == ent.replace(' ', ''):
                    continue
                # filter repeating entity in the entity tuple,
                # e.g., "play" vs "playing"
                if ent.startswith(pred_ent) or pred_ent.startswith(ent):
                    continue

            # filter entity appearing in the prompt
            for raw_prompt, _ in weighted_prompts:
                if pred_ent in raw_prompt:
                    continue

            flag = True
            for word in pred_ent.split():
                if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                    flag = False
                    break

            if flag == False:
                continue
            
            for word in pred_ent.split():
                repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

            heapq.heappush(collected_ent_heap, [logprob.item(), pred_ent])
            while len(collected_ent_heap) > Constants.NUMBER_OF_ALL_TUPLES:
                heapq.heappop(collected_ent_heap)
            cur_ent_tuple.append(pred_ent)
        return collected_ent_heap, collected_tuples_heap, cur_logprobs

    def get_ent1_1_1(self, weighted_prompts, collected_tuples_heap, is_bert_model = True):
        # Fint <ENT1>
        mask_logits_total = None

        for raw_prompt, weight in weighted_prompts:
            prompt = raw_prompt
            if is_bert_model == True:

                prompt = prompt.replace(
                    '<ENT1>',"[MASK]")
            else:
                prompt = prompt.replace(
                    '<ENT1>',"<mask>")

            input_text = prompt
            
            mask_logits = self.get_mask_logits(input_text=input_text)

            mask_needed_index = 0
            
            mask_logits = mask_logits[mask_needed_index]

            if mask_logits_total is None:
                mask_logits_total = torch.zeros_like(mask_logits)
            mask_logits_total = mask_logits_total + mask_logits * weight

        mask_logits_total = mask_logits_total / sum(weight for _, weight in weighted_prompts)

        mask_logits_total[self.banned_ids] = -float('inf')
        logprobs = torch.log_softmax(mask_logits_total, dim=-1)
        logprobs, pred_ids = torch.sort(logprobs, descending=True)

        cur_logprobs = []
        collected_ent_heap = []
        cur_ent_tuple = []
        repeat_cnt = {}
        logprob_threshold = float('-inf') if len(collected_tuples_heap) < Constants.NUMBER_OF_ALL_TUPLES \
            else collected_tuples_heap[0][0]
        
        for logprob, pred_id in zip(logprobs, pred_ids):
            min_logprob_upd = min(cur_logprobs + [logprob.item()])
            if len(collected_ent_heap) == Constants.NUMBER_OF_CANDIDATE_PER_ENT  and \
                    min_logprob_upd < collected_ent_heap[0][0]:
                break

            if min_logprob_upd < logprob_threshold:
                break

            if not any([ch.isalpha() for ch in
                        self.tokenizer.decode(pred_id)]):
                continue

            if any([punc in self.tokenizer.decode(pred_id)
                    for punc in string.punctuation]):
                continue

            cur_logprobs = cur_logprobs + [logprob.item()]
            pred_ent = self.tokenizer.decode(pred_id)

            pred_ent = pred_ent.strip().lower()
            if self.check_entity_valid(pred_ent) == False:
                continue
            for ent in cur_ent_tuple:
                # filter repeating entity in the entity tuple,
                # e.g., "grassland" vs "grass land"
                if pred_ent.replace(' ', '') == ent.replace(' ', ''):
                    continue
                # filter repeating entity in the entity tuple,
                # e.g., "play" vs "playing"
                if ent.startswith(pred_ent) or pred_ent.startswith(ent):
                    continue

            # filter entity appearing in the prompt
            for raw_prompt, _ in weighted_prompts:
                if pred_ent in raw_prompt:
                    continue

            flag = True
            for word in pred_ent.split():
                if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                    flag = False
                    break

            if flag == False:
                continue
            
            for word in pred_ent.split():
                repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

            heapq.heappush(collected_ent_heap, [logprob.item(), pred_ent])
            while len(collected_ent_heap) > Constants.NUMBER_OF_ALL_TUPLES:
                heapq.heappop(collected_ent_heap)
            cur_ent_tuple.append(pred_ent)

        
        return collected_ent_heap, collected_tuples_heap, cur_logprobs

    def get_ent_pairs_1_1(self, weighted_prompts, collected_tuples_heap, is_bert_model = True): ## Must define collected_tuples_heap = []
        # Fint <ENT0>
        collected_ent_heap0, collected_tuples_heap, cur_logprobs0 = self.get_ent0_1_1(weighted_prompts=weighted_prompts, collected_tuples_heap=collected_tuples_heap, is_bert_model=is_bert_model)
        collected_ent_heap0.sort(reverse=True)

        flag0 = set()
        flag1 = set()
        cur_ent_tuple = []
        repeat_cnt = {}

        for ent_logprob0, pred_ent0 in tqdm(collected_ent_heap0):
            if pred_ent0 in flag0:
                continue
            else:
                flag0.add(pred_ent0)

            weighted_prompts_upd = []   
            for prompt, weight in weighted_prompts:
                weighted_prompts_upd.append([prompt.replace(f'<ENT0>', pred_ent0), weight])

             # Find <ENT1>
            collected_ent_heap1, collected_tuples_heap, cur_logprobs1 = self.get_ent1_1_1(weighted_prompts=weighted_prompts_upd, collected_tuples_heap=collected_tuples_heap, is_bert_model=is_bert_model)
            collected_ent_heap1.sort(reverse=True)

            for ent_logprob1, pred_ent1 in collected_ent_heap1:
                if pred_ent1 in flag1:
                    continue
                else:
                    flag1.add(pred_ent1)
                
                if pred_ent0 == pred_ent1:
                    continue

                pred = [min(ent_logprob0, ent_logprob1), [pred_ent0, pred_ent1]]

                cur_ent_tuple = [pred_ent0, pred_ent1]
                flag = True
                for ent in cur_ent_tuple:
                    for word in ent.split():
                        if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                            flag = False
                            break

                if flag == False:
                    continue

                heapq.heappush(collected_tuples_heap, pred)
                for ent in cur_ent_tuple:
                    for word in ent.split():
                        repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

                while len(collected_tuples_heap) > Constants.NUMBER_OF_ALL_TUPLES:
                    heap_top = heapq.heappop(collected_tuples_heap)
                    if len(repeat_cnt) > 0:
                        for ent in heap_top[1]:
                            for word in ent.split():
                                if word in repeat_cnt:
                                    repeat_cnt[word] = repeat_cnt[word] - 1
                                
        return collected_tuples_heap

    def get_predid0_1_1(self, weighted_prompts, collected_tuples_heap, is2_2 = False, is_bert_model = True):
        # Fint <ENT0>
        mask_logits_total = None

        for raw_prompt, weight in weighted_prompts:
            prompt = ""
            if is_bert_model == True:
                prompt = raw_prompt.replace(
                    '<ENT0>',"[MASK]")

                prompt = prompt.replace(
                    '<ENT1>',"[MASK]")
            else:
                prompt = raw_prompt.replace(
                    '<ENT0>',"<mask>")

                prompt = prompt.replace(
                    '<ENT1>',"<mask>")

            input_text = prompt
            
            mask_logits = self.get_mask_logits(input_text=input_text)

            mask_needed_index = 0
            if self.is_ent0_before_ent1_raw_prompt(raw_prompt) == False:
                mask_needed_index = 1
                if is2_2 == True:
                    mask_needed_index = 2
            
            mask_logits = mask_logits[mask_needed_index]

            if mask_logits_total is None:
                mask_logits_total = torch.zeros_like(mask_logits)
            mask_logits_total = mask_logits_total + mask_logits * weight

        mask_logits_total = mask_logits_total / sum(weight for _, weight in weighted_prompts)

        mask_logits_total[self.banned_ids] = -float('inf')
        logprobs = torch.log_softmax(mask_logits_total, dim=-1)
        logprobs, pred_ids = torch.sort(logprobs, descending=True)

        cur_logprobs = []
        collected_ent_heap = []
        cur_ent_tuple = []
        repeat_cnt = {}
        logprob_threshold = float('-inf') if len(collected_tuples_heap) < Constants.NUMBER_OF_ALL_TUPLES \
            else collected_tuples_heap[0][0]
        for logprob, pred_id in zip(logprobs, pred_ids):
            min_logprob_upd = min(cur_logprobs + [logprob.item()])
            #cur_logprobs += [logprob.item()]
            if len(collected_ent_heap) == Constants.NUMBER_OF_ALL_TUPLES  and \
                    min_logprob_upd < collected_ent_heap[0][0]:
                break

            if min_logprob_upd < logprob_threshold:
                break

            if not any([ch.isalpha() for ch in
                        self.tokenizer.decode(pred_id)]):
                continue

            if any([punc in self.tokenizer.decode(pred_id)
                    for punc in string.punctuation]):
                continue

            pred_ent = self.tokenizer.decode(pred_id)

            pred_ent = pred_ent.strip().lower()
            if self.check_entity_valid(pred_ent) == False:
                continue
            for ent in cur_ent_tuple:
                # filter repeating entity in the entity tuple,
                # e.g., "grassland" vs "grass land"
                if pred_ent.replace(' ', '') == ent.replace(' ', ''):
                    continue
                # filter repeating entity in the entity tuple,
                # e.g., "play" vs "playing"
                if ent.startswith(pred_ent) or pred_ent.startswith(ent):
                    continue

            # filter entity appearing in the prompt
            for raw_prompt, _ in weighted_prompts:
                if pred_ent in raw_prompt:
                    continue

            flag = True
            for word in pred_ent.split():
                if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                    flag = False
                    break

            if flag == False:
                continue
            
            for word in pred_ent.split():
                repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

            heapq.heappush(collected_ent_heap, [logprob.item(), pred_id])
            while len(collected_ent_heap) > Constants.NUMBER_OF_ALL_TUPLES:
                heapq.heappop(collected_ent_heap)
            cur_ent_tuple.append(pred_ent)
        return collected_ent_heap, collected_tuples_heap, cur_logprobs

    def get_predid1_1_1(self, weighted_prompts, collected_tuples_heap, is2_2 = False, is_bert_model = True):
        # Fint <ENT1>
        mask_logits_total = None

        for raw_prompt, weight in weighted_prompts:
            prompt = raw_prompt
            if is_bert_model == True:
                prompt = prompt.replace(
                    '<ENT1>',"[MASK]")
            else:
                prompt = prompt.replace(
                    '<ENT1>',"<mask>")

            input_text = prompt
            
            mask_logits = self.get_mask_logits(input_text=input_text)

            mask_needed_index = 0
            if is2_2 == True:
                mask_needed_index = 1
            
            mask_logits = mask_logits[mask_needed_index]

            if mask_logits_total is None:
                mask_logits_total = torch.zeros_like(mask_logits)
            mask_logits_total = mask_logits_total + mask_logits * weight

        mask_logits_total = mask_logits_total / sum(weight for _, weight in weighted_prompts)

        mask_logits_total[self.banned_ids] = -float('inf')
        logprobs = torch.log_softmax(mask_logits_total, dim=-1)
        logprobs, pred_ids = torch.sort(logprobs, descending=True)

        cur_logprobs = []
        collected_ent_heap = []
        cur_ent_tuple = []
        repeat_cnt = {}
        logprob_threshold = float('-inf') if len(collected_tuples_heap) < Constants.NUMBER_OF_ALL_TUPLES \
            else collected_tuples_heap[0][0]
        for logprob, pred_id in zip(logprobs, pred_ids):
            min_logprob_upd = min(cur_logprobs + [logprob.item()])
            #cur_logprobs += [logprob.item()]
            if len(collected_ent_heap) == Constants.NUMBER_OF_CANDIDATE_PER_ENT  and \
                    min_logprob_upd < collected_ent_heap[0][0]:
                break

            if min_logprob_upd < logprob_threshold:
                break

            if not any([ch.isalpha() for ch in
                        self.tokenizer.decode(pred_id)]):
                continue

            if any([punc in self.tokenizer.decode(pred_id)
                    for punc in string.punctuation]):
                continue

            pred_ent = self.tokenizer.decode(pred_id)

            pred_ent = pred_ent.strip().lower()
            if self.check_entity_valid(pred_ent) == False:
                continue
            for ent in cur_ent_tuple:
                # filter repeating entity in the entity tuple,
                # e.g., "grassland" vs "grass land"
                if pred_ent.replace(' ', '') == ent.replace(' ', ''):
                    continue
                # filter repeating entity in the entity tuple,
                # e.g., "play" vs "playing"
                if ent.startswith(pred_ent) or pred_ent.startswith(ent):
                    continue

            # filter entity appearing in the prompt
            for raw_prompt, _ in weighted_prompts:
                if pred_ent in raw_prompt:
                    continue

            flag = True
            for word in pred_ent.split():
                if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                    flag = False
                    break

            if flag == False:
                continue
            
            for word in pred_ent.split():
                repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

            heapq.heappush(collected_ent_heap, [logprob.item(), pred_id])
            while len(collected_ent_heap) > Constants.NUMBER_OF_ALL_TUPLES:
                heapq.heappop(collected_ent_heap)
            cur_ent_tuple.append(pred_ent)

        
        return collected_ent_heap, collected_tuples_heap, cur_logprobs

    def get_ent0_2_1(self, weighted_prompts, collected_tuples_heap, is_bert_model = True):
        weighted_prompts_upd = []
        repeat_cnt = {}
        for prompt, weight in weighted_prompts:
            if is_bert_model == True:
                weighted_prompts_upd.append([prompt.replace(f'<ENT0>', "[MASK]<ENT0>"), weight])
            else: 
                weighted_prompts_upd.append([prompt.replace(f'<ENT0>', "<mask><ENT0>"), weight])
        # Fint <ENT0>

        cur_logprobs = []
        logprob_threshold = float('-inf') if len(collected_tuples_heap) < Constants.NUMBER_OF_ALL_TUPLES \
            else collected_tuples_heap[0][0]

        cur_ent_tuple_ent_0 = []
        collected_ent_heap_ent_0 = []
        collected_ent_heap0, collected_tuples_heap, cur_logprobs0 = self.get_predid0_1_1(weighted_prompts=weighted_prompts_upd, collected_tuples_heap=collected_tuples_heap, is_bert_model=is_bert_model)
        collected_ent_heap0.sort(reverse=True)
        for ent_logprob0, pred_ent0 in collected_ent_heap0:
            ## Check array is full or not
            min_logprob_upd = min(cur_logprobs + [ent_logprob0])
            if len(collected_ent_heap_ent_0) == Constants.NUMBER_OF_ALL_TUPLES and \
                    min_logprob_upd < collected_ent_heap_ent_0[0][0]:
                break

            if min_logprob_upd < logprob_threshold:
                break

            if not any([ch.isalpha() for ch in
                        self.tokenizer.decode(pred_ent0)]):
                continue

            if any([punc in self.tokenizer.decode(pred_ent0)
                    for punc in string.punctuation]):
                continue

            weighted_prompts_upd1 = []   
            for prompt, weight in weighted_prompts_upd:
                pred_ent0_word = self.tokenizer.decode(pred_ent0)
                if is_bert_model == True:
                    weighted_prompts_upd1.append([prompt.replace(f'[MASK]', pred_ent0_word), weight])
                else:
                    weighted_prompts_upd1.append([prompt.replace(f'<mask>', pred_ent0_word), weight])

            
            collected_ent_heap0_temp, collected_tuples_heap, cur_logprobs0_temp = \
                self.get_predid0_1_1(weighted_prompts=weighted_prompts_upd1, collected_tuples_heap=collected_tuples_heap, is_bert_model=is_bert_model)
            collected_ent_heap0_temp.sort(reverse=True)

            for ent_logprob0_temp, pred_ent0_temp in collected_ent_heap0_temp:
                min_logprob_upd = min(cur_logprobs + [ent_logprob0, ent_logprob0_temp])
                if len(collected_ent_heap_ent_0) == Constants.NUMBER_OF_ALL_TUPLES and \
                        min_logprob_upd < collected_ent_heap_ent_0[0][0]:
                    break

                if min_logprob_upd < logprob_threshold:
                    break

                if not any([ch.isalpha() for ch in
                            self.tokenizer.decode(pred_ent0_temp)]):
                    continue

                if any([punc in self.tokenizer.decode(pred_ent0_temp)
                        for punc in string.punctuation]):
                    continue

                cur_token_ids = [pred_ent0] + [pred_ent0_temp]
                pred_ent = self.tokenizer.decode(cur_token_ids)
                pred_ent = pred_ent.strip().lower()
                if self.check_entity_valid(pred_ent) == False:
                    continue
                for ent in cur_ent_tuple_ent_0:
                    # filter repeating entity in the entity tuple,
                    # e.g., "grassland" vs "grass land"
                    if pred_ent.replace(' ', '') == ent.replace(' ', ''):
                        continue
                    # filter repeating entity in the entity tuple,
                    # e.g., "play" vs "playing"
                    if ent.startswith(pred_ent) or pred_ent.startswith(ent):
                        continue

                # filter entity appearing in the prompt
                for raw_prompt, _ in weighted_prompts:
                    if pred_ent in raw_prompt:
                        continue

                flag = True
                for word in pred_ent.split():
                    if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                        flag = False
                        break

                if flag == False:
                    continue
                
                for word in pred_ent.split():
                    repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

                heapq.heappush(collected_ent_heap_ent_0, [min([ent_logprob0_temp, ent_logprob0]), pred_ent])
                while len(collected_ent_heap_ent_0) > Constants.NUMBER_OF_ALL_TUPLES:
                    heapq.heappop(collected_ent_heap_ent_0)
                cur_ent_tuple_ent_0.append(pred_ent)

        return collected_ent_heap_ent_0
             
    def get_ent_pairs_2_1(self, weighted_prompts, collected_tuples_heap, is_bert_model = True):
        print("get_ent_pairs_2_1 - finding ent 0")
        
        collected_ents_0 = self.get_ent0_2_1(weighted_prompts=weighted_prompts, collected_tuples_heap=collected_tuples_heap, is_bert_model=is_bert_model)
        collected_ents_0.sort(reverse=True)

        flag0 = set()
        flag1 = set()
        repeat_cnt = {}
        print("get_ent_pairs_2_1 - finding tuples")

        for ent_logprob0, pred_ent0 in tqdm(collected_ents_0):
            if pred_ent0 in flag0:
                continue
            else:
                flag0.add(pred_ent0)

            weighted_prompts_upd = []
            for prompt, weight in weighted_prompts:
                weighted_prompts_upd.append([prompt.replace(f'<ENT0>', pred_ent0), weight])

            collected_ent_heap_1, collected_tuples_heap, cur_logprobs1 = self.get_ent1_1_1(weighted_prompts=weighted_prompts_upd, collected_tuples_heap=collected_tuples_heap, is_bert_model=is_bert_model)
            collected_ent_heap_1.sort(reverse=True)

            # Find <ENT1>
            for ent_logprob1, pred_ent1 in collected_ent_heap_1:
                if pred_ent1 in flag1:
                    continue
                else:
                    flag1.add(pred_ent1)

                if pred_ent0 == pred_ent1:
                    continue

                pred = [min(ent_logprob0, ent_logprob1), [pred_ent0, pred_ent1]]

                cur_ent_tuple = [pred_ent0, pred_ent1]
                flag = True
                for ent in cur_ent_tuple:
                    for word in ent.split():
                        if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                            flag = False
                            break
                if flag == False:
                    continue

                heapq.heappush(collected_tuples_heap, pred)
                for ent in cur_ent_tuple:
                    for word in ent.split():
                        repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

                while len(collected_tuples_heap) > Constants.NUMBER_OF_ALL_TUPLES:
                    heap_top = heapq.heappop(collected_tuples_heap)
                    if len(repeat_cnt) > 0:
                        for ent in heap_top[1]:
                            for word in ent.split():
                                if word in repeat_cnt:
                                    repeat_cnt[word] = repeat_cnt[word] - 1     
        return collected_tuples_heap
    
    def get_ent_from_mask(self, weighted_prompts, collected_tuples_heap):
        mask_logits_total = None
        for raw_prompt, weight in weighted_prompts:
            
            input_text = raw_prompt
            
            mask_logits = self.get_mask_logits(input_text=input_text)

            mask_idx_in_prompt = 0
            
            mask_logits = mask_logits[mask_idx_in_prompt]

            if mask_logits_total is None:
                mask_logits_total = torch.zeros_like(mask_logits)
            mask_logits_total = mask_logits_total + mask_logits * weight

        mask_logits_total = mask_logits_total / sum(
            weight for _, weight in weighted_prompts)

        mask_logits_total[self.banned_ids] = -float('inf')
        logprobs = torch.log_softmax(mask_logits_total, dim=-1)
        logprobs, pred_ids = torch.sort(logprobs, descending=True)


        cur_logprobs = []
        collected_ent_heap = []
        cur_ent_tuple = []
        repeat_cnt = {}
        logprob_threshold = float('-inf') if len(collected_tuples_heap) < Constants.NUMBER_OF_ALL_TUPLES \
            else collected_tuples_heap[0][0]
        for logprob, pred_id in zip(logprobs, pred_ids):
            min_logprob_upd = min(cur_logprobs + [logprob])

            if len(collected_ent_heap) == Constants.NUMBER_OF_CANDIDATE_PER_ENT  and \
                    min_logprob_upd < collected_ent_heap[0][0]:
                break

            if min_logprob_upd < logprob_threshold:
                break

            if not any([ch.isalpha() for ch in
                        self.tokenizer.decode(pred_id)]):
                continue

            if any([punc in self.tokenizer.decode(pred_id)
                    for punc in string.punctuation]):
                continue

            pred_ent = self.tokenizer.decode(pred_id)

            pred_ent = pred_ent.strip().lower()

            if self.check_entity_valid(pred_ent) == False:
                continue
            for ent in cur_ent_tuple:
                # filter repeating entity in the entity tuple,
                # e.g., "grassland" vs "grass land"
                if pred_ent.replace(' ', '') == ent.replace(' ', ''):
                    continue
                # filter repeating entity in the entity tuple,
                # e.g., "play" vs "playing"
                if ent.startswith(pred_ent) or pred_ent.startswith(ent):
                    continue

            # filter entity appearing in the prompt
            for raw_prompt, _ in weighted_prompts:
                if pred_ent in raw_prompt:
                    continue

            flag = True
            for word in pred_ent.split():
                if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                    flag = False
                    break

            if flag == False:
                continue

            for word in pred_ent.split():
                repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

            heapq.heappush(collected_ent_heap, [logprob.item(), pred_id])
            # heapq.heappush(collected_ent_heap, [min(cur_logprobs + [logprob.item()]), pred_id])
            while len(collected_ent_heap) > Constants.NUMBER_OF_ALL_TUPLES:
                heapq.heappop(collected_ent_heap)
            cur_ent_tuple.append(pred_ent)

        
        return collected_ent_heap, collected_tuples_heap, cur_logprobs

    def get_ent1_1_2(self, weighted_prompts, collected_tuples_heap, is_bert_model = True):
        weighted_prompts_upd = []
        for prompt, weight in weighted_prompts:
            if is_bert_model == True:
                weighted_prompts_upd.append([prompt.replace(f'<ENT1>', "[MASK]<ENT1>"), weight])
            else:
                weighted_prompts_upd.append([prompt.replace(f'<ENT1>', "<mask><ENT1>"), weight])
               
        # Fint <ENT1>
        cur_logprobs = []
        logprob_threshold = float('-inf') if len(collected_tuples_heap) < Constants.NUMBER_OF_ALL_TUPLES \
            else collected_tuples_heap[0][0]
        cur_ent_tuple_ent_1 = []
        collected_ent_heap_ent_1 = []
        repeat_cnt = {}
        collected_ent_heap1, collected_tuples_heap, cur_logprobs1 = self.get_predid1_1_1(weighted_prompts=weighted_prompts_upd, collected_tuples_heap=collected_tuples_heap, is_bert_model=is_bert_model)
        collected_ent_heap1.sort(reverse=True)
        for ent_logprob1, pred_ent1 in collected_ent_heap1:
            min_logprob_upd = min(cur_logprobs + [ent_logprob1])

            if len(collected_ent_heap_ent_1) == Constants.NUMBER_OF_ALL_TUPLES and \
                    min_logprob_upd < collected_ent_heap_ent_1[0][0]:
                break

            if min_logprob_upd < logprob_threshold:
                break

            if not any([ch.isalpha() for ch in
                        self.tokenizer.decode(pred_ent1)]):
                continue

            if any([punc in self.tokenizer.decode(pred_ent1)
                    for punc in string.punctuation]):
                continue

            weighted_prompts_upd1 = []   
            for prompt, weight in weighted_prompts_upd:
                pred_ent_word1 = self.tokenizer.decode(pred_ent1)
                if is_bert_model == True:
                    prompt_temp = prompt.replace(f'[MASK]', pred_ent_word1)
                    prompt_temp = prompt_temp.replace(f'<ENT1>','[MASK]')
                else:
                    prompt_temp = prompt.replace(f'<mask>', pred_ent_word1)
                    prompt_temp = prompt_temp.replace(f'<ENT1>', '<mask>')

                weighted_prompts_upd1.append([prompt_temp, weight])
            
            collected_ent_heap1_temp, collected_tuples_heap, cur_logprobs1_temp = \
                self.get_ent_from_mask(weighted_prompts=weighted_prompts_upd1, collected_tuples_heap=collected_tuples_heap)

            collected_ent_heap1_temp.sort(reverse=True)

            for ent_logprob1_temp, pred_ent1_temp in collected_ent_heap1_temp:
                min_logprob_upd = min(cur_logprobs + [ent_logprob1, ent_logprob1_temp])
                if len(collected_ent_heap_ent_1) == Constants.NUMBER_OF_ALL_TUPLES and \
                        min_logprob_upd < collected_ent_heap_ent_1[0][0]:
                    break

                if min_logprob_upd < logprob_threshold:
                    break

                if not any([ch.isalpha() for ch in
                            self.tokenizer.decode(pred_ent1_temp)]):
                    continue

                if any([punc in self.tokenizer.decode(pred_ent1_temp)
                        for punc in string.punctuation]):
                    continue

                cur_token_ids = [pred_ent1] + [pred_ent1_temp]
                pred_ent = self.tokenizer.decode(cur_token_ids)

                pred_ent = pred_ent.strip().lower()

                if self.check_entity_valid(pred_ent) == False:
                    continue
                for ent in cur_ent_tuple_ent_1:
                    # filter repeating entity in the entity tuple,
                    # e.g., "grassland" vs "grass land"
                    if pred_ent.replace(' ', '') == ent.replace(' ', ''):
                        continue
                    # filter repeating entity in the entity tuple,
                    # e.g., "play" vs "playing"
                    if ent.startswith(pred_ent) or pred_ent.startswith(ent):
                        continue

                # filter entity appearing in the prompt
                for raw_prompt, _ in weighted_prompts:
                    if pred_ent in raw_prompt:
                        continue

                flag = True
                for word in pred_ent.split():
                    if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                        flag = False
                        break

                if flag == False:
                    continue

                for word in pred_ent.split():
                    repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

                heapq.heappush(collected_ent_heap_ent_1, [min([ent_logprob1_temp, ent_logprob1]), pred_ent])
                while len(collected_ent_heap_ent_1) > Constants.NUMBER_OF_ALL_TUPLES:
                    heapq.heappop(collected_ent_heap_ent_1)
                cur_ent_tuple_ent_1.append(pred_ent)

        return collected_ent_heap_ent_1

    def get_ent_pairs_1_2(self, weighted_prompts, collected_tuples_heap, is_bert_model = True):
        print("get_ent_pairs_1_2 - finding ent 0")
        collected_ent_heap_0, collected_tuples_heap, cur_logprobs0 = self.get_ent0_1_1(weighted_prompts=weighted_prompts, collected_tuples_heap=collected_tuples_heap, 
                                                                                       is_bert_model=is_bert_model)

        collected_ents_0  = collected_ent_heap_0
        collected_ents_0.sort(reverse=True)
        flag0 = set()
        flag1 = set()
        repeat_cnt = {}
        print("get_ent_pairs_1_2 - finding tuples")
        for ent_logprob0, pred_ent0 in tqdm(collected_ents_0):
            if pred_ent0 in flag0:
                continue
            else:
                flag0.add(pred_ent0)

            weighted_prompts_upd = []
            for prompt, weight in weighted_prompts:
                weighted_prompts_upd.append([prompt.replace(f'<ENT0>', pred_ent0), weight])
            
            collected_ent_heap_1= self.get_ent1_1_2(weighted_prompts=weighted_prompts_upd, collected_tuples_heap=collected_tuples_heap, is_bert_model=is_bert_model)
            collected_ent_heap_1.sort(reverse=True)

            # Find <ENT1>
            for ent_logprob1, pred_ent1 in collected_ent_heap_1:
                if pred_ent1 in flag1:
                    continue
                else:
                    flag1.add(pred_ent1)

                if pred_ent0 == pred_ent1:
                    continue

                pred = [min(ent_logprob0, ent_logprob1), [pred_ent0, pred_ent1]]

                cur_ent_tuple = [pred_ent0, pred_ent1]
                flag = True
                for ent in cur_ent_tuple:
                    for word in ent.split():
                        if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                            flag = False
                            break

                if flag == False:
                    continue

                heapq.heappush(collected_tuples_heap, pred)
                for ent in cur_ent_tuple:
                    for word in ent.split():
                        repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

                while len(collected_tuples_heap) > Constants.NUMBER_OF_ALL_TUPLES:
                    heap_top = heapq.heappop(collected_tuples_heap)
                    if len(repeat_cnt) > 0:
                        for ent in heap_top[1]:
                            for word in ent.split():
                                if word in repeat_cnt:
                                    repeat_cnt[word] = repeat_cnt[word] - 1                                       
        return collected_tuples_heap

    # New 2-2
    def get_ent0_2_2(self, weighted_prompts, collected_tuples_heap, is_bert_model = True):
        cur_ent_tuple_ent_0 = []
        collected_ent_heap_ent_0 = []
        cur_logprobs = []
        logprob_threshold = float('-inf') if len(collected_tuples_heap) < Constants.NUMBER_OF_ALL_TUPLES \
            else collected_tuples_heap[0][0]
        weighted_prompts_upd = []

        for prompt, weight in weighted_prompts:
            if is_bert_model == True:  
                new_prompt = prompt.replace(f'<ENT0>', "[MASK][MASK]")
                new_prompt = new_prompt.replace(f'<ENT1>', "[MASK][MASK]")
            else:
                new_prompt = prompt.replace(f'<ENT0>', "<mask><mask>")
                new_prompt = new_prompt.replace(f'<ENT1>', "<mask><mask>")

            weighted_prompts_upd.append([new_prompt, weight])

        collected_ent_heap_0, collected_tuples_heap, cur_logprobs0 = \
            self.get_predid0_1_1(weighted_prompts=weighted_prompts_upd, collected_tuples_heap=collected_tuples_heap, is2_2=True)
        collected_ent_heap_0.sort(reverse=True)

        repeat_cnt = {}
        for ent_logprob0, pred_ent0 in collected_ent_heap_0:
            min_logprob_upd = min(cur_logprobs + [ent_logprob0])

            if len(collected_ent_heap_ent_0) == Constants.NUMBER_OF_ALL_TUPLES and \
                    min_logprob_upd < collected_ent_heap_ent_0[0][0]:
                break

            if min_logprob_upd < logprob_threshold:
                break

            if not any([ch.isalpha() for ch in
                        self.tokenizer.decode(pred_ent0)]):
                continue

            if any([punc in self.tokenizer.decode(pred_ent0)
                    for punc in string.punctuation]):
                continue

            for prompt, weight in weighted_prompts:
                pred_ent0_word = self.tokenizer.decode(pred_ent0)
                if is_bert_model == True:
                    new_prompt = prompt.replace(f'<ENT0>', pred_ent0_word + "[MASK]")
                    new_prompt = new_prompt.replace(f'<ENT1>', "[MASK][MASK]")
                else:
                    new_prompt = prompt.replace(f'<ENT0>', pred_ent0_word + "<mask>")
                    new_prompt = new_prompt.replace(f'<ENT1>', "<mask><mask>")

                weighted_prompts_upd.append([new_prompt, weight])

            collected_ent_heap_0_temp, collected_tuples_heap, cur_logprobs0_temp = \
            self.get_predid0_1_1(weighted_prompts=weighted_prompts_upd, collected_tuples_heap=collected_tuples_heap, is2_2=True)

            collected_ent_heap_0_temp.sort(reverse=True)

            for ent_logprob0_temp, pred_ent0_temp in collected_ent_heap_0_temp:
                min_logprob_upd = min(cur_logprobs + [ent_logprob0, ent_logprob0_temp])

                if len(collected_ent_heap_ent_0) == Constants.NUMBER_OF_ALL_TUPLES and \
                        min_logprob_upd < collected_ent_heap_ent_0[0][0]:
                    break

                if min_logprob_upd < logprob_threshold:
                    break

                if not any([ch.isalpha() for ch in
                            self.tokenizer.decode(pred_ent0_temp)]):
                    continue

                if any([punc in self.tokenizer.decode(pred_ent0_temp)
                        for punc in string.punctuation]):
                    continue

                cur_token_ids = [pred_ent0] + [pred_ent0_temp]

                pred_ent = self.tokenizer.decode(cur_token_ids)

                pred_ent = pred_ent.strip().lower()

                if self.check_entity_valid(pred_ent) == False:
                    continue
                for ent in cur_ent_tuple_ent_0:
                    # filter repeating entity in the entity tuple,
                    # e.g., "grassland" vs "grass land"
                    if pred_ent.replace(' ', '') == ent.replace(' ', ''):
                        continue
                    # filter repeating entity in the entity tuple,
                    # e.g., "play" vs "playing"
                    if ent.startswith(pred_ent) or pred_ent.startswith(ent):
                        continue

                # filter entity appearing in the prompt
                for raw_prompt, _ in weighted_prompts:
                    if pred_ent in raw_prompt:
                        continue
                flag = True
                for word in pred_ent.split():
                    if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                        flag = False
                        break

                if flag == False:
                    continue

                for word in pred_ent.split():
                    repeat_cnt[word] = repeat_cnt.get(word, 0) + 1
                heapq.heappush(collected_ent_heap_ent_0, [min([ent_logprob0_temp, ent_logprob0]), pred_ent])
                while len(collected_ent_heap_ent_0) > Constants.NUMBER_OF_ALL_TUPLES:
                    heapq.heappop(collected_ent_heap_ent_0)
                cur_ent_tuple_ent_0.append(pred_ent)

        return collected_ent_heap_ent_0

    def get_ent_pairs_2_2(self, weighted_prompts, collected_tuples_heap, is_bert_model = True):

        collected_ent_heap_ent_0 = self.get_ent0_2_2(weighted_prompts=weighted_prompts, collected_tuples_heap=collected_tuples_heap, is_bert_model=is_bert_model)

        collected_ent_heap_ent_0.sort(reverse=True)

        repeat_cnt = {}

        flag0 = set()
        flag1 = set()

        for ent_logprob0, pred_ent0 in tqdm(collected_ent_heap_ent_0):
            if pred_ent0 in flag0:
                continue
            else:
                flag0.add(pred_ent0)

            weighted_prompts_upd = []

            for prompt, weight in weighted_prompts:
                new_prompt = prompt.replace(f'<ENT0>', pred_ent0)
                weighted_prompts_upd.append([new_prompt, weight])
            
            collected_ent_heap_ent_1 = self.get_ent1_1_2(weighted_prompts=weighted_prompts_upd, collected_tuples_heap=collected_tuples_heap, is_bert_model=is_bert_model)
            collected_ent_heap_ent_1.sort(reverse=True)

            for ent_logprob1, pred_ent1 in collected_ent_heap_ent_1:
                if pred_ent1 in flag1:
                    continue
                else:
                    flag1.add(pred_ent1)

                if pred_ent0 == pred_ent1:
                    continue

                cur_ent_tuple = [pred_ent0, pred_ent1]
                pred = [min(ent_logprob0, ent_logprob1), cur_ent_tuple]

                flag = True

                for ent in cur_ent_tuple:
                    for word in ent.split():
                        if repeat_cnt.get(word, 0) + 1 > Constants.MAX_WORD_REPEAT:
                            flag = False
                            break

                if flag == False:
                    continue

                heapq.heappush(collected_tuples_heap, pred)
                for ent in cur_ent_tuple:
                    for word in ent.split():
                        repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

                while len(collected_tuples_heap) > Constants.NUMBER_OF_ALL_TUPLES:
                    heap_top = heapq.heappop(collected_tuples_heap)
                    if len(repeat_cnt) > 0:
                        for ent in heap_top[1]:
                            for word in ent.split():
                                if word in repeat_cnt:
                                    repeat_cnt[word] = repeat_cnt[word] - 1

       
        return collected_tuples_heap
       
    #######
    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def banned_ids(self):
        return self._banned_ids

