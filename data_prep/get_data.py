#%%
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import spacy 
import pickle
import re

## Trim the sentence around a term, either the section before or after
## Similar approach for both, just differently added
def trim(old_sent, pre=True, cutoff=100):
    new_sent = ''
    words = old_sent.split()
    if pre:
        words = reversed(words)
    
    for word in words:
        if pre:
            new_sent = f'{word} {new_sent}'
        else:
            new_sent = f'{new_sent} {word}'

        if len(new_sent) > cutoff:
            break

    return new_sent

## TODO: fixing for date is bad; but leave it for now
def preprocess_data(docs, corpus_name, path):
    nlp = spacy.load("en_core_web_sm")

    sentences = []
    sent_id = 0
    processed = nlp.pipe(docs.content, batch_size=50, 
        n_process=1, disable=["ner", "textcat"])
    for date, doc in tqdm(zip(docs.date, processed), total=len(docs)):
        ## TODO: this is unideal; not splitting on \n
        for sent in doc.sents:
            ## TODO: fix for more than covid-19 
            p_sent = []
            for token in sent:
                t = token.text.lower()
                if t.isalpha() == True:
                    p_sent.append(token.lemma_.lower())
                elif ('covid' in t):
                    p_sent.extend(re.findall(r'^[a-z]+', t))

            # p_sent = [token.lemma_.lower() for token in sent 
            #         if token.text.isalpha() == True or ('covid' in token.text)]
            
            if p_sent == []:
                continue
            sent_id += 1
            sent_info = [sent_id, corpus_name, str(sent), p_sent, date]
            sentences.append(sent_info)
            
    sentences = pd.DataFrame(sentences,
                columns=['sent_id', 'corpus', 'sentence', 'processed_sentence', 'date']
                )
    sentences.set_index('sent_id', inplace=True)
    sentences.to_pickle(path)

## TODO: this should be integrated with other methods below
def pull_from_preprocessed_data(
    data_path, save_path, targets):

    print(f'Results will be saved to {save_path}')
    Path(save_path).mkdir(parents=True, exist_ok=True)

    data = pd.read_pickle(data_path)
    # data['processed_sentence'] = data['processed_sentence'].apply(literal_eval)
    print(f'\nAll Sents: {len(data):,}')
    data.drop_duplicates(subset=['sentence'], inplace=True)
    print(f'All Sents after duplicates removed: {len(data):,}')

    word_indices = {word:0 for word in targets}
    target_sent_ids = []
    non_target_sents = []
    word_index_sents = []
    target_data = []

    for sent_id, row in tqdm(data.iterrows(), total=len(data)):
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

            pre = trim(' '.join(words[:i]))
            post = trim(' '.join(words[i + 1:]), pre=False)

            formatted_sent = (pre, word, post)
            length = len(pre) + len(post)

            target_info = [word_index, word, formatted_sent, length, sent_id]
            target_data.append(target_info)

        word_index_sents.append(word_index_sent)
        target_sent_ids.append(sent_id)

    print(f'\nTarget Sents: {len(target_sent_ids):,}')
    sentence_data = data.loc[target_sent_ids]
    sentence_data['word_index_sentence'] = word_index_sents
    ## TODO: should drop preproc here
    sentence_data.to_pickle(f'{save_path}/target_sentences.pkl')
    print('Target sents saved!\n')

    print(f'Non-Target Sents: {len(non_target_sents):,}')
    with open(f'{save_path}/non_target.pkl', 'wb') as pout:
        pickle.dump(non_target_sents, pout)
    print('Non-target sents saved!\n')

    print(f'Targets found: {len(target_data)}')
    target_data = pd.DataFrame(target_data, columns=['word_index', 'target', 'formatted_sentence', 'length', 'sent_id'])
    target_data.to_pickle(f'{save_path}/target_information.pkl')
    print('Target data saved!')
    # print('\nTarget Counts')
    # print(target_data.target.value_counts())

def parse_sentences(
    corpora_path, non_target_path, corpus_name, 
    targets, word_indices, sent_id_shift, pattern):

    non_target_sents = []
    sentence_data = []
    target_data = []
    sent_id = sent_id_shift

    print(f'\n== Parsing sentences for {corpus_name} ==')
    with open(f'{corpora_path}/{corpus_name}.txt') as fin:
        lines = [line.lower().strip() for line in fin.readlines()]
        print(f'\t{len(lines):,} sentences pulled')
        lines = list(set(lines))
        print(f'\t{len(lines):,} sentences left after duplicates removed')

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

            pre = trim(pre)
            post = trim(post, pre=False)

            formatted_sent = (pre, just_target, post)
            length = len(pre) + len(post)

            target_info = [word_index, just_target, formatted_sent, length, sent_id, corpus_name]
            target_data.append(target_info)

            word_index_sent.append(word_index)

        sentence_info = [sent_id, corpus_name, line, word_index_sent]
        sentence_data.append(sentence_info)
        sent_id += 1

    print(f'\nTarget Sents: {len(sentence_data):,}')
    print(f'Non-Target Sents: {len(non_target_sents):,}')
    with open(f'{non_target_path}/{corpus_name}_non_target.dat', 'wb') as pout:
        pickle.dump(non_target_sents, pout)
    print('Non-Target Saved!')

    return sentence_data, target_data, word_indices, sent_id

#%%
def pull_target_data(
    corpus_targets, corpora_path, 
    subset_path, pattern=r'[a-z]+'): 
    
    ## Setup 
    ## We keep all target information together regardless of the number of corpura, 
    # so word indices are completely unique
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
    target_data = pd.DataFrame(target_data, columns=['word_index', 'target', 'formatted_sentence', 'length', 'sent_id', 'corpus'])
    sentence_data = pd.DataFrame(sentence_data, columns=['sent_id', 'corpus', 'sentence', 'word_index_sentence'])

    print('\nTarget Counts')
    print(target_data.target.value_counts())
    print(f'\nTotal Target Sents: {len(sentence_data):,}')

    return sentence_data, target_data

def save_data(sentence_data, target_data, output_path):
    sentence_data.set_index('sent_id', inplace=True)
    sentence_data.to_pickle(f'{output_path}/target_sentences.pkl')
    print('\nSentence data saved!')

    target_data.set_index('word_index', inplace=True)
    target_data.to_pickle(f'{output_path}/target_information.pkl')

    print('\nTarget data saved!')
# %%
