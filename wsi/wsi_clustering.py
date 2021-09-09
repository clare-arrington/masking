from collections import Counter, defaultdict
from typing import Dict
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

## Goes through sense cluster data to remap and get best fitting sentences
def remap_senses(sense_remapping, clusters):
    remapped_clusters = defaultdict(list)
    for sense_num, cluster in clusters.items():
        remapped_sense = sense_remapping[sense_num]
        remapped_clusters[remapped_sense].extend(cluster)

    clusters = {}
    for sense_num, cluster in enumerate(remapped_clusters.values()):
        clusters[sense_num] = cluster

    return clusters

def perform_clustering(predictions, settings):
    metric = 'cosine'
    method = 'average'
    dists = pdist(predictions, metric=metric)
    Z = linkage(dists, method=method, metric=metric)

    distance_crit = Z[-settings.max_number_senses, 2]

    labels = fcluster(Z, distance_crit,
                      'distance') - 1

    return labels

def cluster_predictions(predictions, settings):
    labels = perform_clustering(predictions, settings)

    ## Count cluster sizes by instances
    initial_clusters = defaultdict(list)  
    for inst_id, label in zip(predictions.index, labels):
        initial_clusters[label].append(inst_id)

    big_senses = [label for label, sents in initial_clusters.items() if len(sents) >= settings.min_sense_instances]

    ## Remap senses
    if settings.min_sense_instances > 0:
        sense_remapping = {}

        ## Find means of sense clusters
        n_senses = np.max(labels) + 1
        sense_means = np.zeros((n_senses, predictions.shape[1]))
        for sense_num, ids in initial_clusters.items():
            cluster_vectors = predictions.loc[ids]
            cluster_center = np.mean(np.array(cluster_vectors), 0)
            sense_means[sense_num] = cluster_center

        dists = cdist(sense_means, sense_means, metric='cosine')
        closest_senses = np.argsort(dists, )[:, ]

        ## Determine closest sense and set remapping if not big
        for sense_idx in range(n_senses):
            for closest_sense in closest_senses[sense_idx]:
                if closest_sense in big_senses:
                    sense_remapping[sense_idx] = closest_sense
                    break

        sense_clusters = remap_senses(sense_remapping, initial_clusters)
        
    return sense_clusters

def cluster_representatives(representatives, settings):
    ## Reformat for clustering
    ids_ordered = list(representatives.keys())
    all_reps = [y for x in ids_ordered for y in representatives[x]]
    n_represent = len(all_reps) // len(ids_ordered)  

    dict_vectorizer = DictVectorizer(sparse=False)
    rep_mat = dict_vectorizer.fit_transform(all_reps)
    rep_vectors = TfidfTransformer(norm=None).fit_transform(rep_mat).todense()

    labels = perform_clustering(rep_vectors, settings)

    ## Count cluster sizes by instances
    initial_clusters = defaultdict(list)  
    instance_senses = {}
    for i, id in enumerate(ids_ordered):
        id_labels = Counter(labels[i * n_represent:
                                   (i + 1) * n_represent])
        instance_senses[id] = id_labels
        main_sense = id_labels.most_common()[0][0]
        initial_clusters[main_sense].append(id)

    big_senses = [label for label, sents in initial_clusters.items() if len(sents) >= settings.min_sense_instances]

    ## Remap senses
    if settings.min_sense_instances > 0:
        sense_remapping = {}

        ## Find means of sense clusters
        n_senses = np.max(labels) + 1
        sense_means = np.zeros((n_senses, rep_vectors.shape[1]))
        for sense_idx in range(n_senses):
            cluster_ids = np.where(labels == sense_idx)
            cluster_center = np.mean(np.array(rep_vectors)[cluster_ids], 0)
            sense_means[sense_idx] = cluster_center

        dists = cdist(sense_means, sense_means, metric='cosine')
        closest_senses = np.argsort(dists, )[:, ]

        ## Determine closest sense and set remapping if not big
        for sense_idx in range(n_senses):
            for closest_sense in closest_senses[sense_idx]:
                if closest_sense in big_senses:
                    sense_remapping[sense_idx] = closest_sense
                    break

        sense_clusters = remap_senses(sense_remapping, initial_clusters)
        
    return sense_clusters
