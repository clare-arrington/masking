#%%
from wsi.lm_bert import LMBert, get_substitutes, trim_predictions
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from wsi.wsi_clustering import cluster_predictions, cluster_representatives

import pathlib
import pickle
import time
import glob

def save_results(data_subset, dataset_name, sense_clusters, output_path, target):
    ## Reformat sentences
    saved_sentences = []
    num_rows = len(data_subset)
    for sense_label, indices in sense_clusters.items():
        data_rows = data_subset[data_subset.word_index.isin(indices)] 

        for pre, _, post in data_rows.formatted_sentence:
            sentence = f'{pre} {target}.{sense_label} {post}'
            saved_sentences.append(sentence)

    with open(f'{output_path}/clusters/{target}_{num_rows}.dat', 'wb') as fout:
            pickle.dump(sense_clusters, fout)

    with open(f'{output_path}/sentences/{target}_{num_rows}.dat', 'wb') as fout:
        pickle.dump(saved_sentences, fout)

    with open(f'{output_path}/summaries/{target}_{num_rows}.txt', 'w+') as fout:
        print(f'=================== {target.capitalize()} ===================\n', file=fout)
        print(f'{len(sense_clusters)} sense(s); {num_rows} sentences', file=fout)
        print(f'\nUsing data from {dataset_name}', file=fout)

        for sense, cluster in sense_clusters.items():

            print(f'\n=================== Sense {sense} ===================', file=fout)

            print(f'{len(cluster)} sentences\n', file=fout)

            print('Example sentences', file=fout)
            data_rows = data_subset[data_subset.word_index.isin(cluster[:20])]
            for pre, targ, post in data_rows.formatted_sentence:
                print(f'\t{pre} *{targ}* {post}\n', file=fout)

#%%
def main(data, dataset_name, output_path, logging_file, targets, 
        cluster_method='hierarchical',
        use_representatives=False, 
        skip_already_run=False,
        subset_num=1000, 
        load_sentence_path=''
        ):

    ## Pull settings from file
    settings = DEFAULT_PARAMS._asdict()
    settings = WSISettings(**settings)

    ## Load base BERT model
    lm = LMBert(settings.cuda_device, settings.bert_model, settings.max_batch_size)

    ## Only paths that needs to be made
    pathlib.Path(f'{output_path}/sentences').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{output_path}/summaries').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{output_path}/clusters').mkdir(parents=True, exist_ok=True)

    if not skip_already_run:
        ## Start the new logging file for this run
        with open(logging_file, 'w') as flog:
            print(dataset_name, file=flog)
            print(f'\n{len(data)} rows loaded', file=flog)
            print(f'{len(targets)} targets loaded\n', file=flog)
            if load_sentence_path != '':
                print(f'Loading same sentence ids as {load_sentence_path}', file=flog)
    else: 
        ## Remove the targets that have already been run
        for row in glob.glob(f'{output_path}/summaries/*.txt'):
            target, etc = row[len(output_path)+1:].split('_')
            if target in targets:
                targets.remove(target)

    if load_sentence_path != '':
        loaded_sentences = {}
        for row in glob.glob(f'{load_sentence_path}/*.dat'):
            target, _ = row[len(load_sentence_path)+1:].split('_')
            
            if target in targets:
                subset_ids = []
                with open(row, 'rb') as fin:
                    clusters = pickle.load(fin)
                    for id, cluster in clusters.items():
                        subset_ids.extend(cluster)

                loaded_sentences[target] = subset_ids

    for n, target in enumerate(targets):
        # break
        print(f'{n} / {len(targets)} : {target}')
        
        if load_sentence_path != '':
            data_subset = loaded_sentences[target]
            data_subset = data[data['word_index'].isin(data_subset)]
            num_rows = len(data_subset)
        else:
            data_subset = data[(data.target == target) & (data.length <= 150)]
            num_rows = min(len(data_subset), subset_num)
            data_subset = data_subset.sample(num_rows)

        ## We don't want to run WSI on a target that is too small
        # if num_rows < 100:
        #     print(f'Skipping {target}')
        #     continue

        with open(logging_file, 'a') as flog:
            print('====================================\n', file=flog)
            print(f'{target.capitalize()} : {num_rows} rows', file=flog)

            print('\tPredicting representatives...')
            start = time.time()
            predictions = lm.predict_sent_substitute_representatives(data_subset, settings)
            end = time.time()
            print(f'\tPredicting took {end - start:.2f} seconds', file=flog)

            if use_representatives:
                reps = get_substitutes(predictions, target, settings)
                print('\tClustering representatives...')
                start = time.time()
                sense_clusters = cluster_representatives(reps, cluster_method, settings)
                end = time.time()
            else:
                print('\tClustering likelihoods...')
                start = time.time()
                predictions = trim_predictions(predictions, target, settings)
                sense_clusters, cluster_centers = cluster_predictions(predictions, cluster_method, settings)
                end = time.time()

            print(f'\tClustering took {end - start:.2f} seconds', file=flog)

            print('\nCluster results')
            for sense, cluster in sense_clusters.items():
                print(f'\t{sense} : {len(cluster)}', file=flog)
                print(f'\t{sense} : {len(cluster)}')
            print(f'{len(sense_clusters)} clusters')

            # print('\tGetting cluster stats...',)
            # stats = get_stats(clusters, max_iter=2500)

        save_results(data_subset, dataset_name, sense_clusters, output_path, target)
        
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
