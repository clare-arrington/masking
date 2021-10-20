#%%
from wsi.lm_bert import LMBert, trim_predictions
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from wsi.wsi_clustering import cluster_predictions

from typing import List
from pathlib import Path
import pandas as pd
from datetime import datetime
from dateutil import tz
import time

def save_results(data, dataset_desc, sense_clusters, output_path):
    ## Reformat sentences
    num_rows = len(data)
    target = data.target.unique()[0]

    cluster_data = []
    for sense_label, indices in sense_clusters.items():
        filter = data.word_index.isin(indices)
        subset_indices = data[filter].index

        sense_subset = data.loc[subset_indices, ['word_index', 'target', 'sent_id']]
        sense_subset['cluster'] = sense_label

        cluster_data.append(sense_subset)
    
    cluster_data = pd.concat(cluster_data)

    with open(f'{output_path}/summaries/{target}_{num_rows}.txt', 'w+') as fout:
        print(f'=================== {target.capitalize()} ===================\n', file=fout)
        print(f'{len(sense_clusters)} sense(s); {num_rows} sentences', file=fout)
        print(f'\nUsing data from {dataset_desc}', file=fout)

        for sense, cluster in sense_clusters.items():

            print(f'\n=================== Sense {sense} ===================', file=fout)
            print(f'{len(cluster)} sentences\n', file=fout)

            print('Example sentences', file=fout)
            data_rows = data[data.word_index.isin(cluster[:20])]
            for pre, targ, post in data_rows.formatted_sentence:
                print(f'\t{pre} *{targ}* {post}\n', file=fout)

    return cluster_data

def pull_rows(data, targets, subset_num):
    target_rows = {}
    for target in targets:
        # data[data.formatted_sentence.apply(lambda s: len(s[0]) == 0 or len(s[2]) == 0)]
        # If the target is the only thing in the sent, we'll get nonsense. 
        data_subset = data[(data.target.isin(target)) & (data.length >= 10)]
        if subset_num:
            num_rows = min(len(data_subset), subset_num)
            data_subset = data_subset.sample(num_rows)
        else: 
            num_rows = len(data_subset)

        target_rows[target[0]] = (data_subset, num_rows)

    return target_rows

def pull_predictions():
    NotImplemented

def convert_to_local(t):
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    t = datetime.utcfromtimestamp(t)
    t = t.replace(tzinfo=from_zone)
    t = t.astimezone(to_zone)

    return datetime.strftime(t, '%H:%M')

#%%
def main(
        data: pd.DataFrame, 
        dataset_desc: str, 
        output_path: str, logging_file: str, 
        targets: List[str],
        subset_num=None, 
        save_preds=True
        ):

    ## Pull settings from file
    settings = DEFAULT_PARAMS._asdict()
    settings = WSISettings(**settings)

    ## Load base BERT model
    lm = LMBert(settings.cuda_device, settings.bert_model, settings.max_batch_size)

    ## Only paths that need to be made
    Path(f'{output_path}/summaries').mkdir(parents=True, exist_ok=True)
    Path(f'{output_path}/predictions').mkdir(parents=True, exist_ok=True)

    ## Start the new logging file for this run
    with open(logging_file, 'w') as flog:
        print(dataset_desc, file=flog)
        print(f'\n{len(data)} rows loaded', file=flog)
        print(f'{len(targets)} targets loaded\n', file=flog)

    target_rows = pull_rows(data, targets, subset_num)
    sense_data = []

    for n, target_alts in enumerate(sorted(targets)):
        # break
        target = target_alts[0]
        print(f'\n{n} / {len(targets)} : {target}')

        data_subset, num_rows = target_rows[target]

        with open(logging_file, 'a') as flog:
            print('====================================\n', file=flog)
            print(f'{target.capitalize()} : {num_rows} rows', file=flog)

            if num_rows >= 100:
                print(f'\tPredicting for {num_rows} rows...')
                start = time.time()
                print(f'\tStart time: {convert_to_local(start)}')
                predictions = lm.predict_sent_substitute_representatives(data_subset, settings)
                end = time.time()
                print(f'\t  Predicting took {end - start:.2f} seconds', file=flog)
                print(f'\t  End time: {convert_to_local(end)}')
                
                if save_preds: 
                    predictions.to_csv(f'{output_path}/predictions/{target}.csv')
                    print(f'\tPredictions saved')

                print('\n\tClustering likelihoods...')
                start = time.time()
                print(f'\t  Start time: {convert_to_local(start)}')
                predictions = trim_predictions(predictions, target_alts, settings)
                sense_clusters, cluster_centers = cluster_predictions(predictions, settings)
                end = time.time()

                print(f'\t  End time: {convert_to_local(end)}')
                print(f'\tClustering took {end - start:.2f} seconds', file=flog)

            else:
                ## We don't want to run WSI on a target that is too small
                ## Mainly it causes problems for the clustering phase 
                print('\tSkipping WSI; not enough rows', file=flog)
                sense_clusters = {
                    '0' : list(data_subset.word_index)
                }

            print('\nCluster results')
            for sense, cluster in sense_clusters.items():
                print(f'\t{sense} : {len(cluster)}', file=flog)
                print(f'\t{sense} : {len(cluster)}')

        sense_info = save_results(data_subset, dataset_desc, sense_clusters, output_path)
        sense_data.append(sense_info)

    sense_data = pd.concat(sense_data)
    sense_data.to_csv(f'{output_path}/target_sense_labels.csv', index=False)

# %%
def create_sense_sentences(sentence_data, output_path):
    target_data = pd.read_csv(f'{output_path}/target_sense_labels.csv')
    target_data.set_index('word_index', inplace=True) ## maybe this default

    sentence_data.word_index_sentence = sentence_data.word_index_sentence.apply(eval)
    sentence_data.set_index('sent_id', inplace=True) ## maybe this default

    sense_sents = []
    for sent_id, row in sentence_data.iterrows():
        sent = row['word_index_sentence']

        sense_sent = []
        for word in sent:
            if '.' not in word:
                sense_sent.append(word)
            else:
                if word in target_data.index:
                    t_row = target_data.loc[word]
                    sense = f'{t_row.target}.{t_row.cluster}'
                    sense_sent.append(sense)

        sense_sents.append([sent_id, sense_sent])

    sense_data = pd.DataFrame(sense_sents, columns=['sent_id', 'sense_sentence'])
    sense_data.set_index('sent_id', inplace=True) 

    sense_data.to_csv(f'{output_path}/sense_sentences.csv')
