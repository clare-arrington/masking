from pathlib import Path
import pandas as pd
from tqdm import tqdm

def get_sentence_data(sentence_path):
    if 'csv' in sentence_path:
        sentence_data = pd.read_csv(sentence_path, usecols=['sent_idx', 'word_idx_sent'])
        sentence_data.word_idx_sent = sentence_data.word_idx_sent.apply(eval)
        # sentence_data.set_index('sent_idx', inplace=True) ## maybe this default
    elif 'pkl' in sentence_path:
        sentence_data = pd.read_pickle(sentence_path)
    return sentence_data

def process_sentences(sentence_data, target_data, targets, ids):
    good_ids = sentence_data.index.intersection(ids)
    print(f'{len(sentence_data):,} sentences loaded')
    print(f'{len(good_ids):,} overlapping sentences')

    if len(good_ids) < len(ids):
        bad_ids = set(ids) - set(good_ids)
        print(f'Removing {len(bad_ids):,} sents from these targets:')
        print(target_data[target_data.sent_idx.isin(bad_ids)].target.unique())
        ids = good_ids
    sentence_data = sentence_data.loc[ids]

    sense_sents = []
    num_bad = 0
    for sent_id, row in tqdm(sentence_data.iterrows(), total=len(sentence_data)):
        sent = row['word_idx_sent']

        sense_sent = []
        add_sent = True
        for word in sent:
            target = word.split('.')[0]
            if '.' not in word:
                sense_sent.append(word)

            elif target not in targets:
                sense_sent.append(target)

            elif word in target_data.index:
                t_row = target_data.loc[word]
                sense = f'{target}.{t_row.cluster}'
                sense_sent.append(sense)

            else:
                print(f'Bad! {sent_id} - {word}')
                num_bad += 1
                add_sent = False
                break

        if add_sent:
            sense_sents.append([sent_id, sense_sent])

    print(f'{num_bad:,} sentences were skipped')

    return sense_sents

def save_sense_sents(sense_sents, output_path, corpus_name):
    print(f'{len(sense_sents):,} sentences modified with senses')

    sense_data = pd.DataFrame(sense_sents, columns=['sent_idx', 'sense_sent'])
    sense_data.set_index('sent_idx', inplace=True) 

    Path(output_path).mkdir(parents=True, exist_ok=True)
    sense_data.to_pickle(f'{output_path}/{corpus_name}_sense_sentences.pkl')

def create_sense_sentences(sentence_path, output_path, corpus_name, slice_max=None):
    target_data = pd.read_pickle(
        f'{output_path}/target_sense_labels.pkl')
    print(f'{len(target_data):,} targets predicted')

    targets = list(target_data.target.unique())
    print(f'{len(targets)} targets selected')
    
    ids = target_data.sent_idx.unique()
    print(f'{len(ids):,} unique sentences with assigned senses')

    if slice_max is None:
        sentence_data = get_sentence_data(sentence_path)
        sense_sents = process_sentences(sentence_data, target_data, targets, ids)
        save_sense_sents(sense_sents, output_path, corpus_name)

    else:
        for slice_num in range(0, slice_max):
            s_path = f'{sentence_path}/slice_{slice_num}/target_sentences.pkl'
            sentence_data = get_sentence_data(s_path)

            o_path = f'{output_path}/slice_{slice_num}'
            print(f'\n==== Slice {slice_num} ====')
            sense_sents = process_sentences(sentence_data, target_data, targets, ids)
            save_sense_sents(sense_sents, o_path, corpus_name)
