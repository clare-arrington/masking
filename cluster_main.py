from wsi.lm_bert import trim_predictions
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from wsi.wsi_clustering import cluster_predictions, find_best_sents, get_cluster_centers, map_other_instances
from log import record_time
from typing import List
from pathlib import Path
import pandas as pd
import pickle

def get_cluster_data(sense_clusters, target_data):
    cluster_data = []
    for sense_label, subset_indices in sense_clusters.items():
        sense_subset = target_data.loc[subset_indices, ['target', 'sent_idx']]
        sense_subset['cluster'] = sense_label        
        cluster_data.append(sense_subset)
    return pd.concat(cluster_data)

def prep_io(targets, output_path, plot_clusters, print_clusters,
     resume_clustering, dataset_desc):
    settings = DEFAULT_PARAMS._asdict()
    settings = WSISettings(**settings)

    logging_file = f'{output_path}/clustering.log'
    Path(f'{output_path}/summaries').mkdir(parents=True, exist_ok=True)
    Path(f'{output_path}/clusters').mkdir(parents=True, exist_ok=True)
    if plot_clusters:
        Path(f'{output_path}/clusters/plots').mkdir(parents=True, exist_ok=True)
    if print_clusters:
        Path(f'{output_path}/clusters/info').mkdir(parents=True, exist_ok=True)

    ## Start the new logging file for this run
    if not resume_clustering:
        with open(logging_file, 'w') as flog:
            print(dataset_desc, file=flog)
        all_sense_data = None
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

    return settings, logging_file, all_sense_data

def make_clusters(
    target_data: pd.DataFrame,
    targets: List[str],
    dataset_desc: str,
    min_sense_size: int,
    output_path: str,
    embed_sents=False,
    resume_clustering: bool = False,
    plot_clusters: bool = False,
    print_clusters: bool = False
    ):

    settings, logging_file, all_sense_data = prep_io(
        targets, output_path, plot_clusters, print_clusters, 
        resume_clustering, dataset_desc)

    sense_data = []
    for n, target_alts in enumerate(sorted(targets)):
        # break
        target = target_alts[0]
        print(f'\n{n+1} / {len(targets)} : {" ".join(target_alts)}')

        ### Get vectors
        if embed_sents:
            with open(f'{output_path}/vectors/{target}.pkl', 'rb') as vp:
                pred_vectors = pickle.load(vp)
                pred_vectors = pd.DataFrame.from_dict(pred_vectors).T
        else:
            pred_vectors = pd.read_pickle(f'{output_path}/predictions/{target}.pkl')
            # print(f'\tPredictions loaded')
            subset_term_ids = trim_predictions(pred_vectors, target_alts, settings.language)
            pred_vectors = pred_vectors[subset_term_ids]

        ### Clustering step ###
        ## Determine what needs to be done based on number of sentences and settings
        use_clustering = len(pred_vectors) >= (min_sense_size * 2) + 25
        use_subset = len(pred_vectors) > settings.subset_num
        if use_clustering:
            record_time('start')
            if use_subset:
                cluster_subset = pred_vectors.sample(settings.subset_num)
            else:
                cluster_subset = pred_vectors
            
            sense_clusters, cluster_centers = cluster_predictions(
                cluster_subset, target_alts, settings, min_sense_size,
                plot_clusters, print_clusters, f'{output_path}/clusters')
            record_time('end')
        else:
            ## We don't cluster a target that is too small
            sense_clusters = {0 : list(pred_vectors.index)}
            cluster_centers = get_cluster_centers(pred_vectors, 1, sense_clusters)   

        if sense_clusters == None:
            continue

        with open(logging_file, 'a') as flog:
            print('====================================\n', file=flog)
            print(f'{target.capitalize()} : {len(pred_vectors)} rows', file=flog)
            if len(target_alts) > 1:
                print(f'Alt form: {target_alts[1]}', file=flog)
            if not use_clustering:
                ## We don't want to cluster a target that is too small
                print('\tSkipping WSI; not enough rows\n', file=flog)

            print('\n\tCluster results')
            print('\n\tCluster results', file=flog)
            for sense, cluster in sense_clusters.items():
                print(f'\t{sense} : {len(cluster)}', file=flog)
                print(f'\t{sense} : {len(cluster)}')

            ## Cluster the remaining 
            if use_subset:
                other_preds = pred_vectors.drop(index=cluster_subset.index)
                sense_clusters = map_other_instances(other_preds, cluster_centers, sense_clusters)

                print('\n\tFinal clusters with all rows')
                print('\n\tFull clusters', file=flog)
                for sense, cluster in sense_clusters.items():
                    print(f'\t{sense} : {len(cluster)}', file=flog)
                    print(f'\t{sense} : {len(cluster)}')

        sense_data.append(get_cluster_data(sense_clusters, target_data))

        ## Save information
        best_sentences = find_best_sents(target_data, pred_vectors, cluster_centers, sense_clusters)
        save_results( dataset_desc, target, 
                      sense_clusters, best_sentences, len(pred_vectors), output_path)

        center_path = f'{output_path}/clusters/{target}.csv'
        centers = pd.DataFrame(cluster_centers, columns=pred_vectors.columns)
        centers.to_csv(center_path)

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
