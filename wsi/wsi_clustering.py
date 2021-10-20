#%%
from collections import Counter, defaultdict
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

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
    dists = pdist(predictions, metric='cosine')
    Z = linkage(dists, method='average', metric='cosine')

    cutoff = min(settings.max_number_senses, len(Z[:,2]))
    distance_crit = Z[-cutoff, 2]

    labels = fcluster(Z, distance_crit, 'distance') - 1

    return labels, None

## TODO: try to make these uniform
## TODO: currently mean not median
def get_cluster_centers(data, n_senses, sense_clusters=None):
    cluster_centers = np.zeros((n_senses, data.shape[1]))
    for sense_num, ids in sense_clusters.items():
        cluster_vectors = data.loc[ids]
        cluster_center = np.mean(np.array(cluster_vectors), 0)
        cluster_centers[sense_num] = cluster_center

    return cluster_centers

#%%
## TODO: these two below could have the whole remapping step in a function i think
def cluster_predictions(predictions, settings):
    labels, cluster_centers = perform_clustering(predictions, settings)
    n_senses = np.max(labels) + 1

    ## Count cluster sizes by instances
    sense_clusters = defaultdict(list)  
    for inst_id, label in zip(predictions.index, labels):
        sense_clusters[label].append(inst_id)

    ## Find means of sense clusters
    cluster_centers = get_cluster_centers(predictions, n_senses, sense_clusters=sense_clusters)

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
        cluster_centers = get_cluster_centers(predictions, n_senses, sense_clusters=sense_clusters)
        
        ## 
        if n_senses == 1:
            break
        else:
            min_instances += 10
            big_senses = [sense for sense, cluster in sense_clusters.items() if len(cluster) >= min_instances]

    return sense_clusters, cluster_centers
