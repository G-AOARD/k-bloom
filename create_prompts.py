import fire
import json
import nltk
import torch
from nltk import sent_tokenize
nltk.download('stopwords')
from thefuzz import fuzz
from scipy.special import softmax
from data_utils_folder.data_utils import get_n_ents, get_sent, fix_prompt_style
import nltk

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rake_nltk import Rake

from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from lexsubgen import SubstituteGenerator

import itertools
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, set_seed
# from keybert import KeyBERT


import sys
from pathlib import Path
LEXSUBGEN_ROOT = str(Path().resolve())

if LEXSUBGEN_ROOT not in sys.path:
    sys.path.insert(0, LEXSUBGEN_ROOT)

CONFIGS_PATH = Path().resolve() / "configs"
print()

# Loading substitute generator
sg = SubstituteGenerator.from_config(
    str(CONFIGS_PATH / "subst_generators" / "lexsub" / "xlnet_embs.jsonnet")
)

r = Rake()
nltk.download('punkt')
# Load a pre-trained Word2Vec model (e.g., "word2vec-google-news-300")
# w2v_model = api.load("word2vec-google-news-300")

TRANSFORMATIONS_SENT = [['', ''], ['a ', ''], ['the ', '']]
TRANSFORMATIONS_ENT = [
    ['', ''], ['being', 'is'], ['being', 'are'], ['ing', ''], ['ing', 'e']]

import os                       

def get_all_probability_of_sentence(logits, sentence_length):
    res = []
    for i in range(sentence_length):
        val = torch.max(torch.nn.functional.softmax(logits[i])).detach().numpy().item()
        res.append(val)
    return res

def get_idx_keyword(sent, key):
    for i in range(len(sent[0])):
        if sent[0][i] == key:
            return i
    return -1

def search_prompts_T5_new_approach(init_prompts, seed_ent_tuples, similarity_threshold):

    generated_prompts = []
    for prompt in generated_prompts + init_prompts:
        if len(generated_prompts) > 0:
            continue
        cache = prompt
        if len(generated_prompts) > 3:
            break
        while True:
            if len(generated_prompts) > 3:
                break
            keywords = get_keyword_sentence(prompt)
            words = [prompt.split()]

            for key in keywords:
                target_ids = [get_idx_keyword(words, key)]
                print("words: ", words, "target_ids: ", target_ids)
                substitutes, w2id = sg.generate_substitutes(words, target_ids)

                for sub in substitutes[0]:
                    for ent_tuple in tqdm(seed_ent_tuples):
                        ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]
                        ### Only tuples
                        keyword = ent_tuple
                        # Generate all possible shuffles
                        all_shuffles = list(itertools.permutations(ent_tuple))

                        # Convert the shuffles to lists (optional)
                        all_shuffles = [list(shuffle) for shuffle in all_shuffles]

                        generated_prompts = get_new_sentences_from_permutations(all_shuffles, cache, ent_tuple, similarity_threshold, generated_prompts)
                        
                        ent_tuple.append(sub)
                        # Generate all possible shuffles
                        print("keyword: ", keyword, " ent_tuple: ", ent_tuple, " sub ", sub)
                        all_shuffles = list(itertools.permutations(ent_tuple))

                        # Convert the shuffles to lists (optional)
                        all_shuffles = [list(shuffle) for shuffle in all_shuffles]
                        generated_prompts = get_new_sentences_from_permutations(all_shuffles, cache, ent_tuple, similarity_threshold, generated_prompts)
                    

    generated_prompts = sorted(generated_prompts, key=lambda x: x[1], reverse=True)
    # Get the top 10 elements based on float values
    result_array = [item[0] for item in generated_prompts]

    return result_array

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
set_seed(42)
from keytotext import pipeline
k2t = pipeline("k2t")

tokenizer_T5 = AutoTokenizer.from_pretrained("t5-small")
model_T5 = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def key_to_text_T5(keyword):
    res = k2t(keyword)
    # Tokenize the input sequence
    input_ids = tokenizer_T5.encode(res, return_tensors="pt")
    logits = model_T5.generate(input_ids = input_ids,output_scores=True, return_dict_in_generate=True)['scores']
    # Calculate the entropy from the predicted logits
    n_gram_probabilities = get_all_probability_of_sentence(logits, len(logits))
    entropy = -np.sum(n_gram_probabilities * np.log(n_gram_probabilities), axis=-1)
    return res, entropy

def get_new_sentences_from_permutations(all_shuffles, original_prompt, ent_tuple, similarity_threshold, generated_prompts):

    for shuffle in all_shuffles:
        print("shuffle: ", shuffle)
        paraphrased_sentence, entropy = key_to_text_T5(shuffle)
        print("new sent: ", paraphrased_sentence, " - with entropy: ", entropy)
        if "\"" in paraphrased_sentence or "|" in paraphrased_sentence or "'t" in paraphrased_sentence or "-" in paraphrased_sentence:
            continue
        if fuzz.ratio(paraphrased_sentence, original_prompt) >= similarity_threshold:
            continue

        if ent_tuple[0] in paraphrased_sentence and ent_tuple[1] in paraphrased_sentence:
            new_raw_prompt = paraphrased_sentence.replace(ent_tuple[0],"<ENT0>")
            new_raw_prompt = new_raw_prompt.replace(ent_tuple[1],"<ENT1>")
            total_score = calculate_comprehensive_score(original_prompt, paraphrased_sentence, entropy)
        
            if check_new_raw_prompt_valid(new_raw_prompt):
                if new_raw_prompt not in generated_prompts:
                    if len(generated_prompts) > 0:
                        if (max([fuzz.ratio(new_raw_prompt, prompt[0]) for prompt in generated_prompts]) < similarity_threshold):
                            generated_prompts.append([new_raw_prompt, total_score])
                    else:
                        generated_prompts.append([new_raw_prompt, total_score])
    return generated_prompts

def check_new_raw_prompt_valid(new_raw_prompt):
    if new_raw_prompt.count("<") > 2 or new_raw_prompt.count(">") > 2:
        return False
    if (new_raw_prompt.count("<ENT1>") > 1 or new_raw_prompt.count("<ENT0>") > 1):
        return False
    return True

def calculate_comprehensive_score(initial_sentence, paraphrased_sentence, entropy):

    P, R, F1 = score([initial_sentence], [paraphrased_sentence], lang='en', verbose=True)
    bert_score = F1.detach().numpy()[0]

    bleu = sentence_bleu([initial_sentence.split()], paraphrased_sentence.split())

    inverse_bleu = 1.0 / (bleu + 0.01)

    total_score = bert_score * inverse_bleu * entropy

    return total_score

def get_substring_between_two(substring, start, end):
    # Find the starting index of the first substring
    start_index = substring.find(start)
    if start_index == -1:
        return None

    # Find the starting index of the second substring after the first one
    end_index = substring.find(end, start_index + len(start))
    if end_index == -1:
        return None

    # Extract the substring between the two substrings
    result = substring[start_index + len(start):end_index]
    return result

def get_keyword_sentence(sentence):
    r.extract_keywords_from_text(sentence)
    raw_phrases = r.get_ranked_phrases()
    ordered_phrases = []
    for i in range(len(raw_phrases)):
        if not (raw_phrases[i] == "," or raw_phrases[i] == "." or raw_phrases[i] == "!" or raw_phrases[i].lower() == "ent1" or raw_phrases[i].lower() == "ent0"):
            ordered_phrases.append(raw_phrases[i])
    
    print("get keyword: ", ordered_phrases)
    for i in range(len(ordered_phrases) - 1):
        for j in range(i + 1, len(ordered_phrases)):
            idx_i = sentence.find(ordered_phrases[i])
            idx_j = sentence.find(ordered_phrases[j])
            if idx_i > idx_j:
                temp = ordered_phrases[i]
                ordered_phrases[i] = ordered_phrases[j]
                ordered_phrases[j] = temp

    return ordered_phrases

# Tokenize and convert words to vectors
def sentence_to_vectors(sentence, model):
    tokens = sentence.lower().split()
    vectors = []
    for token in tokens:
        if token in model:
            vectors.append(model[token])
    return vectors

def search_prompts_T5_new_approach(init_prompts, seed_ent_tuples, similarity_threshold):

    generated_prompts = []
    for prompt in generated_prompts + init_prompts:
        if len(generated_prompts) > 0:
            continue
        cache = prompt
        if len(generated_prompts) > 3:
            break
        while True:
            if len(generated_prompts) > 3:
                break
            keywords = get_keyword_sentence(prompt)
            words = [prompt.split()]

            for key in keywords:
                target_ids = [get_idx_keyword(words, key)]
                print("words: ", words, "target_ids: ", target_ids)
                substitutes, w2id = sg.generate_substitutes(words, target_ids)

                for sub in substitutes[0]:
                    for ent_tuple in tqdm(seed_ent_tuples):
                        ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]
                        ### Only tuples
                        keyword = ent_tuple
                        # Generate all possible shuffles
                        all_shuffles = list(itertools.permutations(ent_tuple))

                        # Convert the shuffles to lists (optional)
                        all_shuffles = [list(shuffle) for shuffle in all_shuffles]

                        generated_prompts = get_new_sentences_from_permutations(all_shuffles, cache, ent_tuple, similarity_threshold, generated_prompts)
                        
                        ent_tuple.append(sub)
                        # Generate all possible shuffles
                        all_shuffles = list(itertools.permutations(ent_tuple))

                        # Convert the shuffles to lists (optional)
                        all_shuffles = [list(shuffle) for shuffle in all_shuffles]
                        generated_prompts = get_new_sentences_from_permutations(all_shuffles, cache, ent_tuple, similarity_threshold, generated_prompts)
                    

    generated_prompts = sorted(generated_prompts, key=lambda x: x[1], reverse=True)
    # Get the top 10 elements based on float values
    result_array = [item[0] for item in generated_prompts]

    return result_array


from tqdm import tqdm
def main(rel_set='conceptnet', similarity_threshold=65):
    print('------------------------ MAIN SEARCH PROMPT --------------------------')
    relation_info = json.load(open(f'relation_info/{rel_set}.json'))
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for rel, info in tqdm(relation_info.items()):
        info['init_prompts'] = [
            fix_prompt_style(prompt) for prompt in info['init_prompts']]

        if 'prompts' not in info or len(info['prompts']) == 0:
            info['prompts'] = info['init_prompts']
            output_path = f'relation_info/{rel_set}.json'
            json.dump(relation_info, open(output_path, 'w'), indent=4)

            seed_pairs = info['seed_ent_tuples']
            info['prompts'] = search_prompts_T5_new_approach(
                init_prompts=info['init_prompts'],
                seed_ent_tuples=seed_pairs,
                similarity_threshold=similarity_threshold)

            for key, value in info.items():
                print(f'{key}: {value}')
            for prompt in info['prompts']:
                print(f'- {prompt}')
            print('=' * 50)

        output_path = f'relation_info/{rel_set}.json'
        json.dump(relation_info, open(output_path, 'w'), indent=4)
    print('------------------------ END OF MAIN SEARCH PROMPT --------------------------')

if __name__ == '__main__':
    fire.Fire(main)
