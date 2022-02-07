#%%
from wsi.lm_bert import LMBert, trim_predictions
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from wsi.wsi_clustering import cluster_predictions, find_best_sents, get_cluster_centers
from typing import List
from datetime import datetime
from dateutil import tz
from pathlib import Path
import pandas as pd
from glob import glob
import time

# 
def aggregate_data(path, file_name, range_max=6):
    data = []
    for slice_num in range(0,range_max):
        slice_data = pd.read_pickle(f'{path}/slice_{slice_num}/{file_name}.pkl')
        data.append(slice_data)

    return pd.concat(data)

# TODO: I added corpus name to the target data; rerun for other datasets
# base_count - requires target to show up n times or its cut out
# min length - requires sentence to be above length k, important for context window
# occurence limit - this flag makes it so that one sentence doesn't have too many targets in it
# TODO: occurence limit could be handled better if you only remove sentences with targets that are very common

def filter_target_data(
    target_path=None, target_data=None, corpus_name=None, 
    occurence_limit=10, minimum_length=25, base_count=50
    ):
    
    if target_data is None:
        if 'csv' in target_path:
            target_data = pd.read_csv(target_path)
        elif 'pkl' in target_path:
            target_data = pd.read_pickle(target_path)
    
    print(f'{len(target_data):,} target instances pulled')

    if corpus_name is not None:
        target_data = target_data[target_data.corpus == corpus_name]
        print(f'{len(target_data):,} instances within {corpus_name}')

    vc = target_data.target.value_counts()
    og_vc = len(vc)
    print(f'=== {og_vc} targets before anything removed ===')

    if minimum_length:
        ids = target_data[target_data.length <= 25].sent_id.unique()
        target_data = target_data[~target_data.sent_id.isin(ids)]
        print(f'{len(target_data):,} instances after {minimum_length} length minimum applied')

    if occurence_limit:
        vc = target_data.sent_id.value_counts()
        ids = vc[vc <= occurence_limit].index
        target_data = target_data[target_data.sent_id.isin(ids)]
        print(f'{len(target_data):,} after {occurence_limit} occurence limit applied')

    vc = target_data.target.value_counts()
    targets = vc[vc >= base_count].index
    target_data = target_data[target_data.target.isin(targets)]
    new_vc = len(vc)
    print(f'{len(target_data):,} after insufficient targets removed')
    print(f'=== {new_vc} targets left after filtering; {og_vc - new_vc} were removed ===')

    return target_data

def pull_rows(data, subset_num):
    target_rows = {}
    sent_ids = set()
    vc = data.target.value_counts(ascending=True)
    for target in vc.index:        
        # If the target is the only thing in the sent, we'll get nonsense. 
        data_subset = data[(data.target == target)]
        # print(target, len(data_subset))

        ## If too big, completely skip for now. 
        if subset_num is not None and (len(data_subset) > subset_num):
            continue
        else:
            ## Save the ids of all samples in case they overlap. 
            ## Since one sentence can have multiple targets, we want to include them all.

            target_rows[target] = data_subset
            # before = len(sent_ids)
            sent_ids.update(data_subset.sent_id)  
            # print(f'\t{len(sent_ids) - before} ids added')

    ## Now we go back through and resample those that were too big
    ## now that all the samples have been accounted for.
    for target in vc.index:
        if target not in target_rows:
            # print(target)

            data_subset = data[(data.target == target) & (data.length >= 25)]
            already_sampled = sum(data_subset.sent_id.isin(sent_ids))
            # print(target)
            # print(f'\t{already_sampled} already sampled')

            sample_subset = data_subset[~data_subset.sent_id.isin(sent_ids)]
            sample_num = max(0, subset_num - already_sampled)
            sent_ids.update(sample_subset.sample(sample_num).index)
            # print(sample_num)

            target_rows[target] = data_subset[data_subset.sent_id.isin(sent_ids)]
            # print(target, len(data_subset))

    return target_rows

def pull_corpus_rows(data, targets, subset_num):
    target_rows = {}
    for target in targets:
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
        
        target_rows[target[0]] = data_subset

    return target_rows

def convert_to_local(t):
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    t = datetime.utcfromtimestamp(t)
    t = t.replace(tzinfo=from_zone)
    t = t.astimezone(to_zone)

    return datetime.strftime(t, '%H:%M')

def record_time(desc):
    t = convert_to_local(time.time())
    t_str = f'\t  {desc.capitalize()} time : {t}'
    print(t_str)
    
    return t_str

def make_predictions(
    target_data: pd.DataFrame,
    targets: List[str],
    dataset_desc: str,
    output_path: str,
    subset_num=None,
    resume_predicting=False
    ):

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
            print(f'\n{len(target_data):,} rows loaded', file=flog)
            print(f'{len(targets)} targets loaded\n', file=flog)
    else:
        already_predicted = glob(f'{output_path}/predictions/*.pkl')
        skip_targets = [path.split('/')[-1][:-4] for path in already_predicted]
        print(f'{len(skip_targets)} targets already predicted')

        remove_targets = []
        for target in targets:
            if target[0] in skip_targets:
                remove_targets.append(target)
        print(f'Removing {len(remove_targets)} targets')

        for target in remove_targets:
            targets.remove(target)
        print(f'{len(targets)} targets going to be clustered')

    print(f'\nPulling target rows for prediction')
    target_rows = pull_rows(target_data.reset_index(), subset_num)

    for n, target_alts in enumerate(sorted(targets)):
        # break
        target = target_alts[0]
        print(f'\n{n+1} / {len(targets)} : {" ".join(target_alts)}')

        data_subset = target_rows[target]
        num_rows = len(data_subset)

        with open(logging_file, 'a') as flog:
            print('====================================\n', file=flog)
            print(f'{target.capitalize()} : {num_rows} rows', file=flog)
            if len(target_alts) > 1:
                print(f'Alt form: {target_alts[1]}', file=flog)

            print(f'\tPredicting for {num_rows} rows...')
            print('\n' + record_time('start'), file=flog)
            predictions = lm.predict_sent_substitute_representatives(data_subset, settings)
            print(record_time('end') + '\n', file=flog)
            
            predictions.to_pickle(f'{output_path}/predictions/{target}.pkl')
            print(f'\tPredictions saved')

#%%
def make_clusters(
    target_data: pd.DataFrame,
    targets: List[str],
    dataset_desc: str,
    output_path: str,
    resume_clustering: bool = False,
    plot_clusters: bool = False
    ):

    settings = DEFAULT_PARAMS._asdict()
    settings = WSISettings(**settings)

    logging_file = f'{output_path}/clustering.log'

    Path(f'{output_path}/summaries').mkdir(parents=True, exist_ok=True)
    if plot_clusters:
        Path(f'{output_path}/plots').mkdir(parents=True, exist_ok=True)

    ## Start the new logging file for this run
    if not resume_clustering:
        with open(logging_file, 'w') as flog:
            print(dataset_desc, file=flog)
    ## TODO: should I be saving sense sents at every step?
    ## Otherwise this isn't saving any data incrementally
    else:
        all_sense_data = pd.read_pickle(f'{output_path}/target_sense_labels.pkl')
        skip_targets = all_sense_data.target.unique()
        print(f'{len(skip_targets)} targets already clustered')
        
        remove_targets = []
        for target in targets:
            if target[0] in skip_targets:
                remove_targets.append(target)
        print(f'Removing {len(remove_targets)} targets')

        for target in remove_targets:
            targets.remove(target)

        print(f'{len(targets)} targets going to be clustered')

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
                print('\n' + record_time('start'), file=flog)
                sense_clusters, cluster_centers = cluster_predictions(
                    predictions, target_alts, settings, plot_clusters, f'{output_path}/plots')
                print(record_time('end') + '\n', file=flog)
            else:
                ## We don't want to cluster a target that is too small
                print('\tSkipping WSI; not enough rows\n', file=flog)
                sense_clusters = {0 : list(predictions.index)}
                cluster_centers = get_cluster_centers(predictions, 1, sense_clusters)

            print('\n\tCluster results')
            for sense, cluster in sense_clusters.items():
                print(f'\t{sense} : {len(cluster)}', file=flog)
                print(f'\t{sense} : {len(cluster)}')

        cluster_data = []
        for sense_label, subset_indices in sense_clusters.items():
            sense_subset = target_data.loc[subset_indices, ['target', 'sent_id']]
            sense_subset['cluster'] = sense_label        
            cluster_data.append(sense_subset)
        sense_data.append(pd.concat(cluster_data))

        best_sentences = find_best_sents(target_data, predictions, cluster_centers, sense_clusters)
        save_results( dataset_desc, target, 
                      sense_clusters, best_sentences, len(predictions), output_path)

    if len(sense_data) > 0:
        sense_data = pd.concat(sense_data)
        if resume_clustering:
            sense_data = pd.concat([all_sense_data, sense_data])        
        
        sense_data.to_pickle(f'{output_path}/target_sense_labels.pkl')
    else:
        print('Error; nothing was generated')
# %%
def save_results(
    dataset_desc, target, sense_clusters,
    best_sentences, sentence_count, output_path):
    
    with open(f'{output_path}/summaries/{target}.txt', 'w+') as fout:
        print(f'=================== {target.capitalize()} ===================\n', file=fout)
        print(f'{len(best_sentences)} sense(s)', file=fout)
        print(f'{sentence_count} sentences', file=fout)
        print(f'\nUsing data from {dataset_desc}', file=fout)

        for sense, cluster in best_sentences.items():
            print(f'\n=================== Sense {sense} ===================', file=fout)
            print(f'{len(sense_clusters[sense])} sentences\n', file=fout)

            print('Central most sentences', file=fout)
            for index, (pre, targ, post) in cluster:
                print(f'\t{index}', file=fout)
                print(f'\t\t{pre} *{targ}* {post}\n', file=fout)
