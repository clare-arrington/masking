#%%
from wsi.lm_bert import LMBert, get_substitutes, trim_predictions
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from wsi.wsi_clustering import cluster_predictions, cluster_representatives

from pathlib import Path
import pandas as pd
import pickle
import time
import glob

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

def pull_rows(data, targets, load_sentence_path, loaded_sentences, subset_num):
    target_rows = {}
    for target in targets:
        if load_sentence_path != '':
            data_subset = loaded_sentences[target]
            data_subset = data[data['word_index'].isin(data_subset)]
            num_rows = len(data_subset)
        else:
            # data[data.formatted_sentence.apply(lambda s: len(s[0]) == 0 or len(s[2]) == 0)]
            # If the target is the only thing in the sent, we'll get nonsense
            data_subset = data[(data.target == target) & (data.length >= 10)]
            if subset_num:
                num_rows = min(len(data_subset), subset_num)
                data_subset = data_subset.sample(num_rows)
            else: 
                num_rows = len(data_subset)

        target_rows[target] = (data_subset, num_rows)

    return target_rows

#%%
def main(
        data, dataset_desc, 
        output_path, logging_file, targets, 
        cluster_method='hierarchical',
        use_representatives=False, 
        subset_num=None, 
        load_sentence_path=''
        ):

    ## Pull settings from file
    settings = DEFAULT_PARAMS._asdict()
    settings = WSISettings(**settings)

    ## Load base BERT model
    lm = LMBert(settings.cuda_device, settings.bert_model, settings.max_batch_size)

    ## Only paths that needs to be made
    Path(f'{output_path}/summaries').mkdir(parents=True, exist_ok=True)

    ## Start the new logging file for this run
    with open(logging_file, 'w') as flog:
        print(dataset_desc, file=flog)
        print(f'\n{len(data)} rows loaded', file=flog)
        print(f'{len(targets)} targets loaded\n', file=flog)
        if load_sentence_path != '':
            print(f'Loading same sentence ids as {load_sentence_path}', file=flog)

    ## TODO: merge with pull data func
    loaded_sentences = {}
    if load_sentence_path != '':
        for row in glob.glob(f'{load_sentence_path}/*.dat'):
            target, _ = row[len(load_sentence_path)+1:].split('_')
            
            if target in targets:
                subset_ids = []
                with open(row, 'rb') as fin:
                    clusters = pickle.load(fin)
                    for id, cluster in clusters.items():
                        subset_ids.extend(cluster)

                loaded_sentences[target] = subset_ids

    target_rows = pull_rows(data, targets, load_sentence_path, loaded_sentences, subset_num)
    sense_data = []

    for n, target in enumerate(sorted(targets)):
        # break
        print(f'\n{n} / {len(targets)} : {target}')

        data_subset, num_rows = target_rows[target]

        with open(logging_file, 'a') as flog:
            print('====================================\n', file=flog)
            print(f'{target.capitalize()} : {num_rows} rows', file=flog)

            if num_rows >= 100:
                print(f'\tPredicting representatives for {num_rows} rows...')
                start = time.time()
                predictions = lm.predict_sent_substitute_representatives(data_subset, settings)
                end = time.time()
                print(f'\tPredicting took {end - start:.2f} seconds', file=flog)

                if use_representatives:
                    reps = get_substitutes(predictions, target, settings)
                    print('\tClustering substitutes...')
                    start = time.time()
                    sense_clusters, cluster_centers = cluster_representatives(reps, cluster_method, settings)
                    end = time.time()
                else:
                    print('\tClustering likelihoods...')
                    start = time.time()
                    predictions = trim_predictions(predictions, target, settings)
                    sense_clusters, cluster_centers = cluster_predictions(predictions, cluster_method, settings)
                    end = time.time()

                print(f'\tClustering took {end - start:.2f} seconds', file=flog)

                # print(f'{len(sense_clusters)} clusters')

                # print('\tGetting cluster stats...',)
                # stats = get_stats(clusters, max_iter=2500)
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

#%%    
def write_stats(stats, data_subset, best_sentences, senses, fout):
    for label, info in stats.items():

        print(f'\n=================== Sense {label} ===================', file=fout)

        print(f'{len(senses[label])} sentences\n', file=fout)
        print(f'{info[0]} representatives\n', file=fout)

        print('Best Words', file=fout)
        #print(f'\t{", ".join(info[1])}', file=fout)
        print(f'\t{", ".join(info[2])}\n', file=fout)

        print('Best Sentences', file=fout)
        indices = [t[0] for t in best_sentences[label]]
        data_rows = data_subset[data_subset.word_index.isin(indices)]
        for sentence in data_rows.sentence:
            print(f'\t{sentence}\n', file=fout)

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
