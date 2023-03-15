#%%
from collections import Counter, defaultdict
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
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

def plot_clustered_preds(preds, labels, target_alts, path):
    pca = PCA(n_components=2).fit(preds)
    preds_comps = pd.DataFrame(pca.transform(preds), columns=['x', 'y'], index=preds.index)
    preds_comps['size'] = 12

    if type(labels) == dict:
        labels = preds_comps.index.map(labels)
    
    preds_comps['cluster'] = labels    
    preds_comps.sort_values(by='cluster', inplace=True)
    preds_comps['cluster'] = preds_comps['cluster'].astype('category')

    layout = {
        "paper_bgcolor": "#FAFAFA",
        "plot_bgcolor": "#DDDDDD",
        "dragmode": "pan",
        'font': {
            'family': "Courier New, monospace",
            'size': 13
        },
        'margin': {
            'l': 60,
            'r': 40,
            'b': 40,
            't': 40,
            'pad': 4
        },
        'xaxis': {
            "showgrid": True,
            "zeroline": False,
            "visible": True,
            "title": ''
        },
        'yaxis': {
            "showgrid": True,
            "zeroline": False,
            "visible": True,
            "title": ''
        },
        'legend': {
            "title":'Cluster'
        }
        }

    title = f"Clusters for {' and '.join(target_alts)}"
    colors = px.colors.qualitative.Prism + [
        'rgb(136, 204, 238)',
        'rgb(102, 17, 0)',
        'rgb(184, 46, 46)',
        'rgb(13, 42, 99)'
        ]

    fig = px.scatter(
        preds_comps, x='x', y='y', color='cluster', size='size',
        hover_name=preds_comps.index,
        title=title, 
        color_discrete_sequence=colors)
    fig.update_layout(**layout)
    fig.update_traces(
        textposition='top center',
        textfont={'family': "Raleway, sans-serif" }
        )
    # fig.show()
    fig.write_html(path)

def perform_clustering(predictions, settings, method='ward'):
    ## Pairwise distances
    dists = pdist(predictions, metric='euclidean')

    ## Hierarchical agglomerative clustering
    Z = linkage(dists, method=method, metric='euclidean')

    # plt.figure(figsize=(10,6))
    # dn = dendrogram(Z, truncate_mode='lastp', p=15)

    cutoff = min(settings.init_num_senses, len(Z[:,2]))
    distance_crit = Z[-cutoff, 2]
    labels = fcluster(Z, distance_crit, 'distance') - 1

    return labels

def get_cluster_centers(data, n_senses, sense_clusters=None):
    cluster_centers = np.zeros((n_senses, data.shape[1]))
    for sense_num, ids in sense_clusters.items():
        cluster_vectors = data.loc[ids]
        cluster_center = np.median(np.array(cluster_vectors), 0)
        cluster_centers[sense_num] = cluster_center

    return cluster_centers

#%%
def cluster_predictions(
    predictions, target_alts, settings, 
    min_sense_size, plot_clusters, print_clusters, save_path=None):
    labels = perform_clustering(predictions, settings)
    n_senses = np.max(labels) + 1

    ## Export information about the starting cluster formation
    if save_path:
        if plot_clusters:
            init_path = f'{save_path}/plots/{target_alts[0]}_initial_clusters.html'
            plot_clustered_preds(predictions, labels, 
                                target_alts, init_path)  
        if print_clusters:
            init_path = f'{save_path}/info/{target_alts[0]}_initial_clusters.txt'
            with open(init_path, 'w') as f:
                for label, count in Counter(labels).items():
                    print(f'{label}: {count}', file=f)

    ## Count cluster sizes by instances
    sense_clusters = defaultdict(list)  
    for inst_id, label in zip(predictions.index, labels):
        sense_clusters[label].append(inst_id)
    # for i in range(15):
    #     print(i, ':', len(sense_clusters[i]))

    ## Find center (median) of sense clusters
    cluster_centers = get_cluster_centers(
        predictions, n_senses, sense_clusters)

    ## Sets might have many small clusters instead of any big
    ## So we can iteratively get big
    big_senses = []
    for label, count in Counter(labels).items(): 
        if count >= min_sense_size:
            big_senses.append(label)
        
    ## Remap senses if they aren't all big 
    while (len(big_senses) != n_senses): 
        # print('\nbig:', big_senses)
        # print(len(big_senses), 'big ?=', n_senses, 'total')

        ## 2 left but they aren't both big enough, so they'll get merged to 1
        if n_senses == 2:
            big_senses = [0]
            sense_remapping = {0:0,1:0}
        else:
            dists = cdist(cluster_centers, cluster_centers, metric='euclidean')
            closest_senses = np.argsort(dists, )[:, ]

            sense_remapping = {}
            for sense_idx in range(n_senses):
                for closest_sense in closest_senses[sense_idx]:
                    if closest_sense in big_senses:
                        sense_remapping[sense_idx] = closest_sense
                        break

            # print('mapping => ', sense_remapping)
            # print('=============\n')

        sense_clusters = remap_senses(sense_remapping, sense_clusters)
        n_senses = len(sense_clusters)
        cluster_centers = get_cluster_centers(
            predictions, n_senses, sense_clusters)
        # print(cluster_centers.shape)
        
        ## 
        # if n_senses > 1:
        big_senses = []
        for sense, cluster in sense_clusters.items():
            # print(sense, ' : ', len(cluster))
            if len(cluster) >= min_sense_size:
                big_senses.append(sense)
        # else:
        #     break

    if plot_clusters and save_path:
        labels = { sent_id:clust for clust, sents in sense_clusters.items() 
                            for sent_id in sents}

        final_path = f'{save_path}/{target_alts[0]}_final_clusters.html'
        plot_clustered_preds(predictions, labels, target_alts, final_path)

    return sense_clusters, cluster_centers  

def find_best_sents(target_data, predictions, cluster_centers, sense_clusters): 
    best_sents = {}
    for sense, sentences in sense_clusters.items():
        center = cluster_centers[sense]
        preds = predictions.loc[sentences]
        dists = cdist([center], preds, metric='euclidean')
        dist_df = pd.DataFrame( dists[0], 
                                columns=['dist'], 
                                index=preds.index)
        central = dist_df.nsmallest(25, columns=['dist'])
        data_rows = target_data.loc[central.index]
        best_sents[sense] = data_rows.formatted_sent.iteritems()
    return best_sents

def map_other_instances(other_preds, cluster_centers, sense_clusters):
    dists = cdist(cluster_centers, other_preds, metric='euclidean')
    closest_senses = dists.T.argmin(axis=1)
    for sense in sense_clusters.keys():
        rows = other_preds.loc[closest_senses == sense]
        sense_clusters[sense].extend(rows.index)
    return sense_clusters

# %%
