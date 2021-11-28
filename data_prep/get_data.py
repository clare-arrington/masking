#%%
from ast import literal_eval
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import spacy 
import pickle
import re

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

def preprocess_data(docs, corpus_name, path):
    nlp = spacy.load("en_core_web_sm")

    sentences = []
    sent_id = 0
    processed = nlp.pipe(docs, batch_size=50, 
        n_process=1, disable=["ner", "textcat"])
    for doc in tqdm(processed):
        for sent in doc.sents:
            p_sent = [token.lemma_ for token in sent if token.text.isalpha() == True]
            if p_sent == []:
                continue
            sent_id += 1
            sent_info = [sent_id, corpus_name, str(sent), p_sent]
            sentences.append(sent_info)

    sentences = pd.DataFrame(sentences,
                columns=['sent_id', 'corpus', 'sentence', 'processed_sentence'])
    sentences.to_pickle(path, index=False)

def pull_from_preprocessed_data(data_path, save_path, targets):

    print(f'Results will be saved to {save_path}')
    Path(save_path).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    data.set_index('sent_id', inplace=True)
    data['processed_sentence'] = data['processed_sentence'].apply(literal_eval)
    print(f'\nAll Sents: {len(data)}')

    word_indices = {word:0 for word in targets}
    target_sent_ids = []
    non_target_sents = []
    word_index_sents = []
    target_data = []

    for sent_id, row in tqdm(data.iterrows()):
        words = row.processed_sentence
        found_targets = set(targets).intersection(set(words))
        if found_targets == set():
            sent = ' '.join(words)
            non_target_sents.append(sent)
            continue
        
        word_index_sent = []
        for i, word in enumerate(words):
            if word not in found_targets:
                word_index_sent.append(word)
                continue

            index = word_indices[word]
            word_indices[word] += 1

            word_index = f'{word}.{index}'
            word_index_sent.append(word_index)

            pre = trim_pre(' '.join(words[:i]))
            post = trim_post(' '.join(words[i + 1:]))

            formatted_sent = (pre, word, post)
            length = len(pre) + len(post)

            target_info = [word_index, word, formatted_sent, length, sent_id]
            target_data.append(target_info)

        word_index_sents.append(word_index_sent)
        target_sent_ids.append(sent_id)

    print(f'Target Sents: {len(target_sent_ids)}')
    sentence_data = data.loc[target_sent_ids]
    sentence_data['word_index_sentence'] = word_index_sents
    sentence_data.to_pickle(f'{save_path}/target_sentences.pkl')
    print('Target sents saved!\n')

    print(f'Non-Target Sents: {len(non_target_sents)}')
    with open(f'{save_path}/non_target.pkl', 'wb') as pout:
        pickle.dump(non_target_sents, pout)
    print('Non-target sents saved!\n')

    print(f'Targets found: {len(target_data)}')
    print('\nTarget Counts')
    target_data = pd.DataFrame(target_data, columns=['word_index', 'target', 'formatted_sentence', 'length', 'sent_id'])
    target_data.to_pickle(f'{save_path}/target_information.pkl')
    # print(target_data.target.value_counts())
    print('Target data saved!')

def parse_sentences(
    corpora_path, non_target_path, corpus_name, 
    targets, word_indices, sent_id_shift, pattern):

    non_target_sents = []
    sentence_data = []
    target_data = []
    sent_id = sent_id_shift

    print(f'Parsing sentences for {corpus_name.upper()}')
    with open(f'{corpora_path}/{corpus_name}.txt') as fin:
        lines = [line.lower().strip() for line in fin.readlines()]
        print(f'\t{len(lines)} sentences pulled')
        lines = list(set(lines))
        print(f'\t{len(lines)} sentences left after duplicates removed')

    for line in tqdm(lines):

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
def pull_target_data(corpus_targets, corpora_path, subset_path, pattern=r'[a-z]+'): 
    ## Setup 
    all_targets = [target for targets in corpus_targets.values() for target in targets]
    word_indices = {word:0 for word in set(all_targets)}
    sent_id_shift = 0
    sentence_data = []
    target_data = []

    pattern = re.compile(pattern)

    print(f'Results will be saved to {subset_path}')
    Path(subset_path).mkdir(parents=True, exist_ok=True)

    for corpus_name, targets in corpus_targets.items():
        s_data, t_data, word_indices, sent_id_shift = \
            parse_sentences(corpora_path, subset_path, corpus_name,
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
