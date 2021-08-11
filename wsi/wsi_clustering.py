from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.svm import LinearSVC

## Goes through sense cluster data to remap and get best fitting sentences
def remap_senses(labels, sense_remapping, instance_senses, num_sents):
    new_order_of_senses = list(set(sense_remapping.values()))
    sense_remapping = dict((k, new_order_of_senses.index(v)) for k, v in sense_remapping.items())
    labels = np.array([sense_remapping[x] for x in labels])

    best_sentences = {sense : [] for sense in set(labels)}
    senses = {sense : [] for sense in set(labels)}

    for inst_id, inst_id_clusters in instance_senses.items():
        for sense, count in inst_id_clusters.most_common(1):
            if sense_remapping:
                sense = sense_remapping[sense]
                
            senses[sense].append(inst_id)

            if len(best_sentences[sense]) < num_sents:
                best_sentences[sense].append((inst_id, count))
            else:
                smallest_tup = min(best_sentences[sense], key = lambda t: t[1])
                if count > smallest_tup[1]:
                    best_sentences[sense].remove(smallest_tup)
                    best_sentences[sense].append((inst_id, count))

    return labels, senses, best_sentences

# TODO: comment
def perform_clustering(representatives, settings):
    dict_vectorizer = DictVectorizer(sparse=False)
    rep_mat = dict_vectorizer.fit_transform(representatives)
    if settings.disable_tfidf:
        transformed = rep_mat
    else:
        transformed = TfidfTransformer(norm=None).fit_transform(rep_mat).todense()

    metric = 'cosine'
    method = 'average'
    dists = pdist(transformed, metric=metric)
    Z = linkage(dists, method=method, metric=metric)

    distance_crit = Z[-settings.max_number_senses, 2]

    labels = fcluster(Z, distance_crit,
                      'distance') - 1

    return labels, rep_mat, transformed, dict_vectorizer

def cluster_inst_ids_representatives(inst_ids_to_representatives, settings, num_sents=10):
    """
    preforms agglomerative clustering on representatives of one SemEval target
    :param inst_ids_to_representatives: map from SemEval instance id to list of representatives
    :param n_clusters: fixed number of clusters to use
    :param disable_tfidf: disable tfidf processing of feature words
    :return: map from SemEval instance id to soft membership of clusters and their weight
    """
    ## Reformat for clustering
    inst_ids_ordered = list(inst_ids_to_representatives.keys())
    representatives = [y for x in inst_ids_ordered for y in inst_ids_to_representatives[x]]
    n_represent = len(representatives) // len(inst_ids_ordered)

    ## Cluster reps
    labels, rep_mat, transformed, dict_vectorizer = perform_clustering(representatives, settings)

    ## Count cluster sizes by instances
    senses_n_domminates = Counter()
    instance_senses = {}
    for i, inst_id in enumerate(inst_ids_ordered):
        inst_id_clusters = Counter(labels[i * n_represent:
                                          (i + 1) * n_represent])
        instance_senses[inst_id] = inst_id_clusters
        senses_n_domminates[inst_id_clusters.most_common()[0][0]] += 1

    big_senses = [x for x in senses_n_domminates if senses_n_domminates[x] >= settings.min_sense_instances]

    ## Remap senses
    if settings.min_sense_instances > 0:
        sense_remapping = {}

        ## Find means of sense clusters
        n_senses = np.max(labels) + 1
        sense_means = np.zeros((n_senses, transformed.shape[1]))
        for sense_idx in range(n_senses):
            idxs_this_sense = np.where(labels == sense_idx)
            cluster_center = np.mean(np.array(transformed)[idxs_this_sense], 0)
            sense_means[sense_idx] = cluster_center

        dists = cdist(sense_means, sense_means, metric='cosine')
        closest_senses = np.argsort(dists, )[:, ]

        ## Determine closest sense and set remapping if not big
        for sense_idx in range(n_senses):
            for closest_sense in closest_senses[sense_idx]:
                if closest_sense in big_senses:
                    sense_remapping[sense_idx] = closest_sense
                    break

        labels, senses, best_sentences = remap_senses(labels, sense_remapping, instance_senses, num_sents)
        
    stat_vars = (labels, rep_mat, transformed, dict_vectorizer)
    return stat_vars, senses, best_sentences, senses_n_domminates, big_senses

def get_stats(labels, rep_mat, transformed, dict_vectorizer, max_iter=1000, dual=False):
    label_count = Counter(labels)
    statistics = {}
    if len(label_count) > 1:
        svm = LinearSVC(class_weight='balanced', penalty='l1', dual=dual, max_iter=max_iter)
        svm.fit(rep_mat, labels)

        coefs = svm.coef_
        top_coefs = np.argpartition(coefs, -10)[:, -10:]
        if top_coefs.shape[0] == 1:
            top_coefs = [top_coefs[0], -top_coefs[0]]

        rep_arr = np.asarray(rep_mat)
        totals_cols = rep_arr.sum(0)

        p_feats = totals_cols / transformed.shape[0]

        for sense_idx, top_coef_sense in enumerate(top_coefs):
            count_reps = label_count[sense_idx]
            count_sense_feat = rep_arr[np.where(labels == sense_idx)]
            p_sense_feat = count_sense_feat.sum(0) / transformed.shape[0]

            pmis_proxy = p_sense_feat / (p_feats + 0.00000001)
            best_features_pmi_idx = np.argpartition(pmis_proxy, -10)[-10:]

            best_features = [dict_vectorizer.feature_names_[x] for x in top_coef_sense]
            best_features_pmi = [dict_vectorizer.feature_names_[x] for x in best_features_pmi_idx]

            statistics[sense_idx] = (count_reps, best_features, best_features_pmi)
    else:
        sense_idx = 0
        rep_arr = np.asarray(rep_mat)
        totals_cols = rep_arr.sum(0)

        p_feats = totals_cols / transformed.shape[0]

        count_reps = label_count[sense_idx]
        count_sense_feat = rep_arr[np.where(labels == sense_idx)]
        p_sense_feat = count_sense_feat.sum(0) / transformed.shape[0]

        pmis_proxy = p_sense_feat / (p_feats + 0.00000001)
        best_features_pmi_idx = np.argpartition(pmis_proxy, -10)[-10:]

        best_features_pmi = [dict_vectorizer.feature_names_[x] for x in best_features_pmi_idx]

        statistics[sense_idx] = (count_reps, [], best_features_pmi)

    return statistics
