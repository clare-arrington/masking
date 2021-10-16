#%%
from collections import Counter, defaultdict
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AffinityPropagation
import numpy as np

def remap_senses(sense_remapping, clusters):
    remapped_clusters = defaultdict(list)
    for sense_num, cluster in clusters.items():
        remapped_sense = sense_remapping[sense_num] ## TODO: sometimes this breaks with a key error; why?
        remapped_clusters[remapped_sense].extend(cluster)

    clusters = {}
    for sense_num, cluster in enumerate(remapped_clusters.values()):
        clusters[sense_num] = cluster

    return clusters

def perform_clustering(predictions, cluster_method, settings):
    if cluster_method == 'affinity_propagation':
        ## If AP doesn't converge, try it again and up the 
        cluster_centers = []
        iters = 0
        while len(cluster_centers) == 0   :
            iters += 1000 
            ## TODO: may have to adjust preference, but leaving alone for now
            af = AffinityPropagation(damping=.9,  max_iter=iters, verbose=True).fit(predictions)
            cluster_centers = af.cluster_centers_ # TODO: could be indices?
            
        labels = af.labels_
        return labels, cluster_centers

    else:
        dists = pdist(predictions, metric='cosine')
        Z = linkage(dists, method='average', metric='cosine')

        cutoff = min(settings.max_number_senses, len(Z[:,2]))
        distance_crit = Z[-cutoff, 2]

        labels = fcluster(Z, distance_crit, 'distance') - 1

        return labels, None

## TODO: try to make these uniform
## TODO: currently mean not median
def get_cluster_centers(data_type, data, n_senses, sense_clusters=None, labels=None):
    if data_type == 'likelihoods':
        cluster_centers = np.zeros((n_senses, data.shape[1]))
        for sense_num, ids in sense_clusters.items():
            cluster_vectors = data.loc[ids]
            cluster_center = np.mean(np.array(cluster_vectors), 0)
            cluster_centers[sense_num] = cluster_center

    elif data_type == 'representatives': 
        cluster_centers = np.zeros((n_senses, data.shape[1]))
        for sense_idx in range(n_senses):
            cluster_ids = np.where(labels == sense_idx)
            cluster_center = np.mean(np.array(data)[cluster_ids], 0)
            cluster_centers[sense_idx] = cluster_center

    return cluster_centers

#%%
## TODO: these two below could have the whole remapping step in a function i think
def cluster_predictions(predictions, cluster_method, settings):
    labels, cluster_centers = perform_clustering(predictions, cluster_method, settings)
    n_senses = np.max(labels) + 1

    ## Count cluster sizes by instances
    sense_clusters = defaultdict(list)  
    for inst_id, label in zip(predictions.index, labels):
        sense_clusters[label].append(inst_id)

    ## Find means of sense clusters
    if cluster_centers is None:
        cluster_centers = get_cluster_centers('likelihoods', predictions, n_senses, sense_clusters=sense_clusters)

    ## Sets might have many small clusters instead of any big
    ## So we can iteratively get big
    ## TODO: could fix this loop up a bit
    big_senses = [label for label, count in Counter(labels).items() if count >= settings.min_sense_instances]
    
    min_instances = 10
    if len(big_senses) == 0:
        big_senses = [label for label, count in Counter(labels).items() if count >= min_instances]
        
    ## Remap senses if they aren't all big
    while (len(big_senses) != n_senses or min_instances <= 50): 
        
        if n_senses == 2:
            big_senses = [0]

        dists = cdist(cluster_centers, cluster_centers, metric='cosine')
        closest_senses = np.argsort(dists, )[:, ]

        ## Determine closest sense and set remapping if not big
        sense_remapping = {}
        for sense_idx in range(n_senses):
            for closest_sense in closest_senses[sense_idx]:
                if closest_sense in big_senses:
                    sense_remapping[sense_idx] = closest_sense
                    break

        # print(n_senses, big_senses)
        # print(sense_remapping)

        sense_clusters = remap_senses(sense_remapping, sense_clusters)
        n_senses = len(sense_clusters)
        cluster_centers = get_cluster_centers('likelihoods', predictions, n_senses, sense_clusters=sense_clusters)
        
        ## 
        if n_senses == 1:
            break
        else:
            min_instances += 10
            big_senses = [sense for sense, cluster in sense_clusters.items() if len(cluster) >= min_instances]

    return sense_clusters, cluster_centers

## Just a mess
## TODO: didn't edit for AP, assuming I'll remove anyway?
def cluster_representatives(representatives, cluster_method, settings):
    ## Reformat for clustering
    ids_ordered = list(representatives.keys())
    all_reps = [y for x in ids_ordered for y in representatives[x]]
    n_represent = len(all_reps) // len(ids_ordered)  

    dict_vectorizer = DictVectorizer(sparse=False)
    rep_mat = dict_vectorizer.fit_transform(all_reps)
    rep_vectors = TfidfTransformer(norm=None).fit_transform(rep_mat).todense()

    labels, _ = perform_clustering(rep_vectors, cluster_method, settings)
    n_senses = np.max(labels) + 1

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

    min_instances = 10
    if len(big_senses) == 0:
        big_senses = [label for label, sents in initial_clusters.items() if len(sents) >= min_instances]
        
    ## Remap senses if they aren't all big
    while len(big_senses) != n_senses: 

        ## Find means of sense clusters
        cluster_centers = get_cluster_centers('representatives', rep_vectors, n_senses, labels=labels)
        dists = cdist(cluster_centers, cluster_centers, metric='cosine')
        closest_senses = np.argsort(dists, )[:, ]

        ## Determine closest sense and set remapping if not big
        sense_remapping = {}
        for sense_idx in range(n_senses):
            for closest_sense in closest_senses[sense_idx]:
                if closest_sense in big_senses:
                    sense_remapping[sense_idx] = closest_sense
                    break

        sense_clusters = remap_senses(sense_remapping, initial_clusters)
        n_senses = len(sense_clusters)
        cluster_centers = get_cluster_centers('representatives', rep_vectors, n_senses, labels=labels)

        min_instances += 10
        big_senses = [sense for sense, cluster in sense_clusters.items() if len(cluster) >= min_instances]

    return sense_clusters, cluster_centers
