#%%
from os import cpu_count
import pickle
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm

def trim_pre(pre, cutoff=100):
    new_pre = ''
    for word in reversed(pre.split()):
        new_pre = f'{word} {new_pre}'
        if len(new_pre) > cutoff:
            break

    return new_pre

def trim_post(post, cutoff=100):
    new_post = ''
    for word in post.split():
        new_post = f'{new_post} {word}'
        if len(new_post) > cutoff:
            break

    return new_post

# line = 'she keep her head_nn tip_vb down so that her long dark blond hair fall over her face_nn to hide the fact that part_nn of her low jaw be miss'

def parse_sentences(
    corpus_path, non_target_path, corpus_name, 
    targets, word_indices, sent_id_shift, pattern):

    non_target_sents = []
    sentence_data = []
    target_data = []
    sent_id = sent_id_shift

    print(f'Parsing sentences for {corpus_name.upper()}')
    with open(f'{corpus_path}/{corpus_name}.txt') as fin:
        for line in tqdm(fin.readlines()):

            line = line.lower().strip()
            words = re.findall(pattern, line)
            found_targets = set(targets).intersection(set(words))

            fully_cleaned_words = []
            for word in words:
                if word in targets:
                    word = word.split('_')[0]
                fully_cleaned_words.append(word)

            if found_targets == set():
                non_target_sents.append(line.lower())
                continue
            
            word_index_sent = []
            for i, word in enumerate(words):
                if word not in found_targets:
                    word_index_sent.append(word)
                    continue

                index = word_indices[word]
                word_indices[word] += 1

                just_target, *etc = word.split('_')
                word_index = f'{just_target}.{index}'

                pre = ' '.join(fully_cleaned_words[:i])
                post = ' '.join(fully_cleaned_words[i + 1:])

                pre = trim_pre(pre)
                post = trim_post(post)

                formatted_sent = (pre, just_target, post)
                length = len(pre) + len(post)

                target_info = [word_index, just_target, formatted_sent, length, sent_id]
                target_data.append(target_info)

                word_index_sent.append(word_index)

            sentence_info = [sent_id, corpus_name, line, word_index_sent]
            sentence_data.append(sentence_info)
            sent_id += 1

    print(f'\nTarget Sents: {len(sentence_data)}')
    print(f'Non-Target Sents: {len(non_target_sents)}')
    with open(f'{non_target_path}/{corpus_name}_non_target.dat', 'wb') as pout:
        pickle.dump(non_target_sents, pout)

    print('Non-Target Saved!\n')

    ## TODO: is this correct?
    return sentence_data, target_data, word_indices, sent_id

#%%
def pull_target_data(targets, data_path, non_target_path, corpus_names, pattern=r'[a-z]+'): 
    ## Setup 
    word_indices = {word:0 for word in targets}
    sent_id_shift = 0
    sentence_data = []
    target_data = []

    pattern = re.compile(pattern)

    Path(non_target_path).parent.mkdir(parents=True, exist_ok=True)

    for corpus_name in corpus_names:
        s_data, t_data, word_indices, sent_id_shift = \
            parse_sentences(data_path, non_target_path, corpus_name,
            targets, word_indices, sent_id_shift, pattern)
        
        sentence_data.extend(s_data)
        target_data.extend(t_data)

    ## Convert to dataframes
    target_data = pd.DataFrame(target_data, columns=['word_index', 'target', 'formatted_sentence', 'length', 'sent_id'])
    sentence_data = pd.DataFrame(sentence_data, columns=['sent_id', 'corpus', 'sentence', 'word_index_sentence'])

    print('\n\nTarget Counts')
    print(target_data.target.value_counts())
    print(f'Total Target Sents: {len(sentence_data)}')

    return sentence_data, target_data

def save_data(sentence_data, target_data, output_path):
    sentence_data.to_csv(f'{output_path}/target_sentences.csv', index=False)
    target_data.to_csv(f'{output_path}/target_information.csv', index=False)

    print('\nTarget data saved!')
# %%
