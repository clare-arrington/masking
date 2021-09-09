#%%
from numpy.core.numeric import indices
from wsi.lm_bert import LMBert, get_substitutes
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from wsi.wsi_clustering import cluster_predictions, cluster_representatives

import pandas as pd
import pathlib
import pickle
import time
import glob

#%%
## Pull settings from file
settings = DEFAULT_PARAMS._asdict()
settings = WSISettings(**settings)

## Load base BERT model
lm = LMBert(settings.cuda_device, settings.bert_model, settings.max_batch_size)

#%%
## SemEval Info
# num = 2
# input_path = f'../data/semeval/corpora/ccoha{num}_target_sents.csv'
# output_path = f'../data/masking_results/semeval/corpus{num}'
# target_path = '../data/semeval/targets.txt'
# dataset_name = f'SemEval Corpus {num}'

## US Info
wordset = 'dissimilar'
target_path = f'../data/us_uk/truth/{wordset}.txt'
input_path = '../data/us_uk/coca_target_sents.csv'

output_path = '../data/masking_results/us_uk/subs'
logging_file = f'{output_path}/{wordset}.log'

## Only path that needs to be made
pathlib.Path(f'{output_path}/sentences').mkdir(parents=True, exist_ok=True)
pathlib.Path(f'{output_path}/summaries').mkdir(parents=True, exist_ok=True)
pathlib.Path(f'{output_path}/clusters').mkdir(parents=True, exist_ok=True)

if wordset == 'similar':
    dataset_name = 'US Corpus - Similar\n(Words have a UK equivalent with similar meaning, i.e. gas = petrol)'
else:
    dataset_name = 'US Corpus - Dissimilar\n(Words have a UK equivalent with different meaning, i.e. football)'

subset_num = 100
use_representatives = True
load_sentences = True
skip_already_run = False

## Pull data
data = pd.read_csv(input_path)
data.formatted_sentence = data.formatted_sentence.apply(eval)

with open(target_path) as fin:
    if wordset == 'similar':
        targets = []
        pairs = fin.read().strip()
        for pair in pairs.split('\n'):
            uk_word, us_word = pair.split()
            targets.append(us_word)
    else:
        targets = fin.read().split()


if skip_already_run:
    for row in glob.glob(f'{output_path}/*.txt'):
        target, etc = row[len(output_path)+1:].split('_')
        ## Remove redundant runs
        if target in targets:
            targets.remove(target)

#%%

if not skip_already_run:
    with open(logging_file, 'w') as flog:
        print(dataset_name, file=flog)
        print(f'\n{len(data)} rows loaded', file=flog)
        print(f'{len(targets)} targets loaded\n', file=flog)

#%%
shift = 0
for n, target in enumerate(targets[shift:]):
    # break
    print(f'{n+shift} / {len(targets)} : {target}')
    #target = target[:-3]

    if skip_already_run:
        count = target_counts[target]
        if count != replace_num:
            print(f'Not replacing {target}, only {count}\n')
            continue
    
    if load_sentences:
        NotImplemented
    else:
        data_subset = data[(data.target == target) & (data.length <= 150)]
        saved_sentences = []

        num_rows = min(len(data_subset), subset_num)
        data_subset = data_subset.sample(num_rows)

    if num_rows < 100:
        print(f'Skipping {target}')
        continue

    ## Do all the main stuff
    with open(logging_file, 'a') as flog:
        print('====================================\n', file=flog)
        print(f'{target.capitalize()} : using {num_rows} rows', file=flog)

        print('\tPredicting representatives...')
        start = time.time()
        predictions = lm.predict_sent_substitute_representatives(data_subset, settings)
        end = time.time()
        print(f'\nPredicting took {end - start:.2f} seconds', file=flog)

        if use_representatives:
            reps = get_substitutes(predictions, settings)
            print('\tClustering representatives...')
            start = time.time()
            sense_clusters = cluster_representatives(reps, settings)
            end = time.time()
        else:
            print('\tClustering likelihoods...')
            start = time.time()
            sense_clusters = cluster_predictions(predictions, settings)
            end = time.time()

        print(f'Clustering took {end - start:.2f} seconds', file=flog)

        for sense, cluster in sense_clusters.items():
            print(f'\t{sense} : {len(cluster)}', file=flog)
            print(f'\t{sense} : {len(cluster)}')
        print(f'{len(sense_clusters)} clusters')

        # print('\tGetting cluster stats...',)
        # stats = get_stats(clusters, max_iter=2500)

    with open(f'{output_path}/clusters/{target}_{num_rows}.dat', 'wb') as fout:
        pickle.dump(sense_clusters, fout)

    ## Reformat sentences
    for sense_label, indices in sense_clusters.items():
        data_rows = data_subset[data_subset.word_index.isin(indices)] 

        for pre, _, post in data_rows.formatted_sentence:
            sentence = f'{pre} {target}.{sense_label} {post}'
            saved_sentences.append(sentence)

    with open(f'{output_path}/sentences/{target}_{num_rows}.dat', 'wb') as fout:
        pickle.dump(saved_sentences, fout)

    with open(f'{output_path}/summaries/{target}_{num_rows}.txt', 'w+') as fout:
        print(f'=================== {target.capitalize()} ===================\n', file=fout)
        print(f'{len(sense_clusters)} senses and {num_rows} sentences', file=fout)
        print(f'Using data from {dataset_name}', file=fout)

        for sense, cluster in sense_clusters.items():

            print(f'\n=================== Sense {sense} ===================', file=fout)

            print(f'{len(cluster)} sentences\n', file=fout)

            print('Example sentences', file=fout)
            data_rows = data_subset[data_subset.word_index.isin(cluster[:20])]
            for pre, targ, post in data_rows.formatted_sentence:
                print(f'\t{pre} *{targ}* {post}\n', file=fout)

    break

        # for label, info in stats.items():

        #     print(f'\n=================== Sense {label} ===================', file=fout)

        #     print(f'{len(senses[label])} sentences\n', file=fout)
        #     print(f'{info[0]} representatives\n', file=fout)

        #     print('Best Words', file=fout)
        #     #print(f'\t{", ".join(info[1])}', file=fout)
        #     print(f'\t{", ".join(info[2])}\n', file=fout)

        #     print('Best Sentences', file=fout)
        #     indices = [t[0] for t in best_sentences[label]]
        #     data_rows = data_subset[data_subset.word_index.isin(indices)]
        #     for sentence in data_rows.sentence:
        #         print(f'\t{sentence}\n', file=fout)

# %%


