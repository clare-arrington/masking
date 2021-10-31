#%%
from wsi.lm_bert import LMBert, trim_predictions
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from wsi.wsi_clustering import cluster_predictions

from typing import List
from datetime import datetime
from dateutil import tz
from pathlib import Path
import pandas as pd
from glob import glob
import time

def get_data(sentence_path, target_path, corpus_name=None, limit_occurences=True):
    sentence_data = pd.read_csv(sentence_path, usecols=['corpus', 'sent_id'])
    sentence_data.set_index(['sent_id'], inplace=True)

    target_data = pd.read_csv(target_path)
    target_data = target_data.join(sentence_data, on='sent_id')
    print(f'{len(target_data)} total target instances pulled')

    if corpus_name is not None:
        target_data = target_data[target_data.corpus == corpus_name]
        print(f'{len(target_data)} instances within {corpus_name}')

    ## Only picking rows with 1 target
    # TODO: this isn't ideal; the problem being avoided is making sure that no sentence with mult. targets
    # only gets sense induction for one of them
    if limit_occurences:
        vc = target_data.sent_id.value_counts()
        ids = vc[vc <= 2].index
        target_data = target_data[target_data.sent_id.isin(ids)]
        print(f'{len(target_data)} after limit applied')

    target_data.formatted_sentence = target_data.formatted_sentence.apply(eval)

    return target_data

def pull_rows(data, targets, subset_num):
    target_rows = {}
    for target in targets:
        # data[data.formatted_sentence.apply(lambda s: len(s[0]) == 0 or len(s[2]) == 0)]
        # If the target is the only thing in the sent, we'll get nonsense. 
        data_subset = data[(data.target.isin(target)) & (data.length >= 25)]
        if subset_num:
            samples = []

            for corpus in data_subset.corpus.unique():
                c_subset = data_subset[data_subset.corpus == corpus]
                num_rows = min(len(c_subset), subset_num)
                c_subset = c_subset.sample(num_rows)
                samples.append(c_subset)

            data_subset = pd.concat(samples)
        
        num_rows = len(data_subset)
        target_rows[target[0]] = (data_subset, num_rows)

    return target_rows

def convert_to_local(t):
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    t = datetime.utcfromtimestamp(t)
    t = t.replace(tzinfo=from_zone)
    t = t.astimezone(to_zone)

    return datetime.strftime(t, '%H:%M')

def make_predictions(
    data: pd.DataFrame,
    dataset_desc: str,
    output_path: str,
    targets: List[str],
    subset_num=None,
    resume_predicting=False
):

    ## Pull settings from file
    settings = DEFAULT_PARAMS._asdict()
    settings = WSISettings(**settings)

    ## Load base BERT model
    lm = LMBert(settings.cuda_device, settings.bert_model, settings.max_batch_size)

    Path(f'{output_path}/predictions').mkdir(parents=True, exist_ok=True)
    logging_file = f'{output_path}/prediction.log'

    ## Start the new logging file for this run
    if not resume_predicting:
        with open(logging_file, 'w') as flog:
            print(dataset_desc, file=flog)
            print(f'\n{len(data)} rows loaded', file=flog)
            print(f'{len(targets)} targets loaded\n', file=flog)

    if resume_predicting:
        already_predicted = glob(f'{output_path}/predictions/*.pkl')
        skip_targets = [path.split('/')[-1][:-4] for path in already_predicted]
        
        remove_targets = []
        for target in targets:
            if target[0] in skip_targets:
                remove_targets.append(target)

        for target in remove_targets:
            targets.remove(target)

    target_rows = pull_rows(data, targets, subset_num)

    for n, target_alts in enumerate(sorted(targets)):
        # break
        target = target_alts[0]
        print(f'\n{n+1} / {len(targets)} : {" ".join(target_alts)}')

        data_subset, num_rows = target_rows[target]

        with open(logging_file, 'a') as flog:
            print('====================================\n', file=flog)
            print(f'{target.capitalize()} : {num_rows} rows', file=flog)
            if len(target_alts) > 1:
                print(f'Alt form: {target_alts[1]}', file=flog)

            print(f'\tPredicting for {num_rows} rows...')
            start = time.time()
            print(f'\t  Start time : {convert_to_local(start)}')
            print(f'\n\t  Start time : {convert_to_local(start)}', file=flog)
            predictions = lm.predict_sent_substitute_representatives(data_subset, settings)
            end = time.time()
            print(f'\t    End time : {convert_to_local(end)}\n', file=flog)
            print(f'\t    End time : {convert_to_local(end)}')
            
            predictions.to_pickle(f'{output_path}/predictions/{target}.pkl')
            print(f'\tPredictions saved')

#%%
def perform_clustering(
        data: pd.DataFrame,
        targets: List[str],
        dataset_desc: str, 
        output_path: str
        ):

    ## Pull settings from file
    settings = DEFAULT_PARAMS._asdict()
    settings = WSISettings(**settings)

    ## Only path that needs to be made
    Path(f'{output_path}/summaries').mkdir(parents=True, exist_ok=True)
    logging_file = f'{output_path}/clustering.log'

    # target_paths = glob(f'{output_path}/predictions/*.pkl')

    ## Start the new logging file for this run
    with open(logging_file, 'w') as flog:
        print(dataset_desc, file=flog)

    sense_data = []
    for n, target_alts in enumerate(sorted(targets)):
        # break
        target = target_alts[0]
        print(f'\n{n+1} / {len(targets)} : {" ".join(target_alts)}')

        predictions = pd.read_pickle(f'{output_path}/predictions/{target}.pkl')
        # print(f'\tPredictions loaded')
        predictions = trim_predictions(predictions, target_alts)

        with open(logging_file, 'a') as flog:
            print('====================================\n', file=flog)
            print(f'{target.capitalize()} : {len(predictions)} rows', file=flog)
            if len(target_alts) > 1:
                print(f'Alt form: {target_alts[1]}', file=flog)

            if len(predictions) >= 100:
                # print('\n\tClustering likelihoods...')
                start = time.time()
                print(f'\t  Start time : {convert_to_local(start)}')
                print(f'\t  Start time : {convert_to_local(start)}', file=flog)
                sense_clusters, cluster_centers = cluster_predictions(predictions, settings)
                end = time.time()
                print(f'\t    End time : {convert_to_local(end)}', file=flog)
                print(f'\t    End time : {convert_to_local(end)}')

            else:
                ## We don't want to cluster a target that is too small
                print('\tSkipping WSI; not enough rows\n', file=flog)
                sense_clusters = {
                    '0' : list(predictions.index)
                }

            print('\nCluster results')
            for sense, cluster in sense_clusters.items():
                print(f'\t{sense} : {len(cluster)}', file=flog)
                print(f'\t{sense} : {len(cluster)}')

        sense_info = save_results(data, dataset_desc, target, sense_clusters, output_path)
        sense_data.append(sense_info)

    sense_data = pd.concat(sense_data)
    sense_data.to_csv(f'{output_path}/target_sense_labels.csv', index=False)
# %%
def save_results(data, dataset_desc, target, sense_clusters, output_path):
    cluster_data = []
    for sense_label, indices in sense_clusters.items():
        filter = data.word_index.isin(indices)
        subset_indices = data[filter].index

        sense_subset = data.loc[subset_indices, ['word_index', 'target', 'sent_id']]
        sense_subset['cluster'] = sense_label

        cluster_data.append(sense_subset)
    
    cluster_data = pd.concat(cluster_data)

    with open(f'{output_path}/summaries/{target}.txt', 'w+') as fout:
        print(f'=================== {target.capitalize()} ===================\n', file=fout)
        print(f'{len(sense_clusters)} sense(s); {len(cluster_data)} sentences', file=fout)
        print(f'\nUsing data from {dataset_desc}', file=fout)

        for sense, cluster in sense_clusters.items():
            print(f'\n=================== Sense {sense} ===================', file=fout)
            print(f'{len(cluster)} sentences\n', file=fout)

            print('Example sentences', file=fout)
            data_rows = data[data.word_index.isin(cluster[:20])]
            for pre, targ, post in data_rows.formatted_sentence:
                print(f'\t{pre} *{targ}* {post}\n', file=fout)

    return cluster_data

def create_sense_sentences(sentence_path, output_path):
    target_data = pd.read_csv(f'{output_path}/target_sense_labels.csv')
    target_data.set_index('word_index', inplace=True) ## maybe this default

    sentence_data = pd.read_csv(sentence_path, usecols=['sent_id', 'word_index_sentence'])
    sentence_data.word_index_sentence = sentence_data.word_index_sentence.apply(eval)
    sentence_data.set_index('sent_id', inplace=True) ## maybe this default

    sense_sents = []
    for sent_id, row in sentence_data.iterrows():
        sent = row['word_index_sentence']

        sense_sent = []
        for word in sent:
            if '.' not in word:
                sense_sent.append(word)
            elif word in target_data.index:
                t_row = target_data.loc[word]
                sense = f'{t_row.target}.{t_row.cluster}'
                sense_sent.append(sense)
            else:
                print(f'Bad! {sent_id}')
                break

        sense_sents.append([sent_id, sense_sent])

    sense_data = pd.DataFrame(sense_sents, columns=['sent_id', 'sense_sentence'])
    sense_data.set_index('sent_id', inplace=True) 

    sense_data.to_csv(f'{output_path}/sense_sentences.csv')
