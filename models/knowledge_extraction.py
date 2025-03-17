from tqdm import tqdm
from scipy.special import softmax

from models.language_model_wrapper import LanguageModelWrapper
from models.triple_search import TripleSearch

from data_utils_folder.data_utils import fix_prompt_style, is_valid_prompt
import Constants
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
import torch

class KnowledgeExtraction:
    def __init__(self,
                 model_name,
                 max_n_prompts=20,
                 max_n_ent_tuples=10000,
                 max_word_repeat=5,
                 max_ent_subwords=1,
                 prompt_temp=1.):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._max_n_prompts = max_n_prompts
        self._max_n_ent_tuples = max_n_ent_tuples
        self._max_word_repeat = max_word_repeat
        self._max_ent_subwords = max_ent_subwords
        self._prompt_temp = prompt_temp

        self._initial_prompt = ""

        self._model = LanguageModelWrapper(model_name=model_name)
        self._ent_tuple_searcher = TripleSearch(model=self._model)

        self._seed_ent_tuples = None
        self._tokenizer = self._model.tokenizer
        self.tokenizer_t5 = T5Tokenizer.from_pretrained("t5-large")
        self.model_t5 = T5ForConditionalGeneration.from_pretrained("t5-large")

    def clear(self):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._seed_ent_tuples = None

    def set_seed_ent_tuples(self, seed_ent_tuples):
        self._seed_ent_tuples = seed_ent_tuples

    def set_prompts(self, prompts):
        for prompt in prompts:
            if is_valid_prompt(prompt=prompt):
                self._weighted_prompts.append([fix_prompt_style(prompt), 1.]) # set default score for each prompt

    def update_prompts(self, initial_prompt): # Calculate the compability score with new prompts and initial entity pairs
        print("[knwoledge_harvester] update_prompts with self._weighted_prompts: ", self._weighted_prompts)

        for i, (prompt, _) in enumerate(self._weighted_prompts):
            pos_scores, neg_scores = [], []
            for ent_tuple in self._seed_ent_tuples:

                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]
                pos_scores.append(self.score(initial_prompt=initial_prompt,
                    prompt=prompt, ent_tuple=ent_tuple))

                for ent_idx in range(len(ent_tuple)):
                    for ent_tuple1 in self._seed_ent_tuples:
                        if ent_tuple1[ent_idx] == ent_tuple[ent_idx]:
                            continue

                        ent_tuple_neg = ent_tuple[:ent_idx] + [ent_tuple1[ent_idx]] + ent_tuple[ent_idx + 1:]

                        neg_scores.append(self.score(
                            prompt=prompt, ent_tuple=ent_tuple_neg))
           
            pos_score = (sum(pos_scores) + 1) / (len(pos_scores) + 1)

            neg_score = (sum(neg_scores) + 1) / (len(neg_scores) + 1)

            self._weighted_prompts[i][1] = \
                (pos_score - 0.5 * neg_score) / self._prompt_temp
            
            print("[knwoledge_harvester] update_prompts - pos_score: ", pos_score, " neg_score: ", neg_score, )

        self._weighted_prompts = sorted(
            self._weighted_prompts,
            key=lambda t: t[1], reverse=True)[:self._max_n_prompts]

        norm_weights = softmax([weight for _, weight in self._weighted_prompts])
        norm_weights[norm_weights < 0.05] = 0.
        norm_weights /= norm_weights.sum()

        for i, norm_weight in enumerate(norm_weights):
            self._weighted_prompts[i][1] = norm_weight
        self._weighted_prompts = [
            t for t in self._weighted_prompts if t[1] > 1e-4]

    def update_ent_tuples(self, top_k, initial_prompt, is_bert_model=True):
        self._weighted_ent_tuples = []
        self._initial_prompt = initial_prompt
        self._weighted_ent_tuples  = self._ent_tuple_searcher.search(
            raw_initial_prompt = initial_prompt,
            weighted_prompts=self._weighted_prompts,
            seed_ent_tuples = self._seed_ent_tuples,
            max_word_repeat=self._max_word_repeat,
            max_ent_subwords=self._max_ent_subwords, 
            n=self._max_n_ent_tuples, # max word repeat is max prompt for seaching in each relation,
            top_k=top_k, is_bert_model=is_bert_model)

        print("------------[knwoledge_harvester] update_ent_tuples with size: ", len(self._weighted_ent_tuples))

    def select_ent_tuples(self):
        print("----------[knwoledge_harvester] select_ent_tuples")

        self._weighted_ent_tuples  = self._ent_tuple_searcher.find_appropriate_tuples(
            self._weighted_ent_tuples, self._initial_prompt, self._seed_ent_tuples)
        return self._weighted_ent_tuples

    def score_ent_tuple(self, ent_tuple):
        score = 0.
        for prompt, weight in self.weighted_prompts:
            score += weight * self.score(prompt=prompt, ent_tuple=ent_tuple)
        return score

    def score(self, initial_prompt, prompt, ent_tuple):
        print('[knwoledge_harvester] score: prompt - ', prompt, ' and ent_tuple - ', ent_tuple)
        logprobs = self._model.fill_ent_tuple_in_prompt(
            prompt=prompt, ent_tuple=ent_tuple)['mask_logprobs']

        token_wise_score = sum(logprobs) / len(logprobs)
        ent_wise_score = sum(logprobs) / len(ent_tuple)
        min_score = min(logprobs)

        return (token_wise_score + ent_wise_score + min_score) / 3.

    # NEW APPROACH
    
    def get_all_probability_of_sentence(self, logits, sentence_length):
        res = []
        for i in range(sentence_length):
            val = torch.max(torch.nn.functional.softmax(logits[i])).detach().numpy().item()
            res.append(val)
        return res
    
    def calculate_comprehensive_score(self, initial_prompt, paraphrased_prompt):
        input_ids = self.tokenizer_t5.encode(paraphrased_prompt, return_tensors="pt")

        # Calculate the entropy from the predicted logits
        logits = self.model_t5.generate(input_ids = input_ids,output_scores=True, return_dict_in_generate=True)['scores']
        n_gram_probabilities = self.get_all_probability_of_sentence(logits, len(logits))
        entropy = -np.sum(n_gram_probabilities * np.log(n_gram_probabilities), axis=-1)

        _, _, F1 = score([initial_prompt], [paraphrased_prompt], lang='en', verbose=True)
        bert_score = F1.detach().numpy()[0]

        bleu = sentence_bleu([initial_prompt.split()], paraphrased_prompt.split())

        inverse_bleu = 1.0 / (bleu + 0.01)

        total_score = bert_score * inverse_bleu * entropy

        return total_score
    def score_new_approach(self, initial_prompt, paraphrased_prompt, ent_tuple):

        initial_prompt = initial_prompt.replace("<ENT0>", ent_tuple[0])
        initial_prompt = initial_prompt.replace("<ENT1>", ent_tuple[1])

        paraphrased_prompt = paraphrased_prompt.replace("<ENT0>", ent_tuple[0])
        paraphrased_prompt = paraphrased_prompt.replace("<ENT1>", ent_tuple[1])

        return self.calculate_comprehensive_score(initial_prompt, paraphrased_prompt)

    def update_prompts_new_approach(self, initial_prompt): # Calculate the compability score with new prompts and initial entity pairs

        for i, (prompt, _) in tqdm(enumerate(self._weighted_prompts)):
            print("update_prompts_new_approach with idx: ",i)
            pos_scores, neg_scores = [], []
            for ent_tuple in self._seed_ent_tuples:

                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]
                pos_scores.append(self.score_new_approach(initial_prompt=initial_prompt,
                    paraphrased_prompt=prompt, ent_tuple=ent_tuple))

                for ent_idx in range(len(ent_tuple)):
                    for ent_tuple1 in self._seed_ent_tuples:
                        if ent_tuple1[ent_idx] == ent_tuple[ent_idx]:
                            continue

                        ent_tuple_neg = ent_tuple[:ent_idx] + [ent_tuple1[ent_idx]] + ent_tuple[ent_idx + 1:]

                        neg_scores.append(self.score_new_approach(initial_prompt=initial_prompt,
                            paraphrased_prompt=prompt, ent_tuple=ent_tuple_neg))
           
            pos_score = (sum(pos_scores) + 1) / (len(pos_scores) + 1)

            neg_score = (sum(neg_scores) + 1) / (len(neg_scores) + 1)

            self._weighted_prompts[i][1] = (pos_score - 0.5 * neg_score) / self._prompt_temp
            

        self._weighted_prompts = sorted(
            self._weighted_prompts,
            key=lambda t: t[1], reverse=True)[:self._max_n_prompts]

        norm_weights = softmax([weight for _, weight in self._weighted_prompts])
        norm_weights[norm_weights < 0.05] = 0.
        norm_weights /= norm_weights.sum()

        
        temp_weight_prompt = [ t for t in self._weighted_prompts if t[1] > 1e-4]
        self._weighted_prompts = temp_weight_prompt
    
    def set_weighted_ent_tuples(self, values):
        self._weighted_ent_tuples = values

    def set_initial_prompt(self, initial_prompt):
        self._initial_prompt = initial_prompt[0]

    @property
    def weighted_ent_tuples(self):
        return self._weighted_ent_tuples

    @property
    def weighted_prompts(self):
        return self._weighted_prompts
    
    @property
    def max_n_ent_tuples(self):
        return self._max_n_ent_tuples
    
    @property
    def max_n_prompts(self):
        return self._max_n_prompts

    @property
    def model(self):
        return self._model