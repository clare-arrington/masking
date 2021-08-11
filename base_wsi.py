#%%
from numpy.core.numeric import indices
from wsi.lm_bert import LMBert
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from wsi.wsi_clustering import cluster_inst_ids_representatives, get_stats
import pandas as pd
import pickle
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
wordset = 'similar'
input_path = '../data/us_uk/coca_target_sents.csv'
output_path = '../data/masking_results/us_uk/coca'
target_path = f'../data/us_uk/truth/{wordset}.txt'
logging_file = f'../data/us_uk/{wordset}.log'

if wordset == 'similar':
    dataset_name = 'US Corpus - Similar\n(Words have a UK equivalent with similar meaning, i.e. gas = petrol)'
else:
    dataset_name = 'US Corpus - Dissimilar\n(Words have a UK equivalent with different meaning, i.e. football)'

## Pull data
data = pd.read_csv(input_path)
data.formatted_sentence = data.formatted_sentence.apply(eval)

#%%
with open(target_path) as fin:
    if wordset == 'similar':
        targets = []
        pairs = fin.read().strip()
        for pair in pairs.split('\n'):
            uk_word, us_word = pair.split()
            targets.append(us_word)
    else:
        targets = fin.read().split()

for row in glob.glob(f'{output_path}/*.txt'):
    target, etc = row[len(output_path)+1:].split('_')
    ## Remove redundant runs
    if target in targets:
        targets.remove(target)

with open(logging_file, 'w') as flog:
    print(dataset_name, file=flog)
    print(f'\n{len(data)} rows loaded', file=flog)
    print(f'{len(targets)} targets loaded\n', file=flog)

subset_num = 1000

#%%
shift = 0
for n, target in enumerate(targets[shift:]):
    print(f'{n+shift} / {len(targets)} : {target}')
    #target = target[:-3]

    # count = target_counts[target]
    # if count != replace_num:
        # print(f'Not replacing {target}, only {count}\n')
        # continue
    
    data_subset = data[(data.target == target) & (data.length <= 150)]
    saved_sentences = []

    num_rows = min(len(data_subset), subset_num)
    data_subset = data_subset.sample(num_rows)

    if num_rows < 100:
        print(f'Skipping {target}')
        continue

    ## Do all the main stuff
    with open(logging_file, 'a') as flog:
        print(f'{target.capitalize()} : using {num_rows} rows', file=flog)

        print('\tPredicting representatives...', file=flog)
        reps, predictions = lm.predict_sent_substitute_representatives(data_subset, settings, max_pred=250)

        print('\tClustering representatives...', file=flog)
        stat_input, senses, best_sentences, clusters, big_senses = cluster_inst_ids_representatives(reps, settings, num_sents=20)

        print(f'\t{clusters}', file=flog)
        print(f'\tBig senses : {big_senses}', file=flog)

        print('\tGetting cluster stats...\n', file=flog)
        stats = get_stats(*stat_input, max_iter=2500)

    ## Reformat sentences
    for sense_label, indices in senses.items():
        data_rows = data_subset[data_subset.word_index.isin(indices)]

        for pre, _, post in data_rows.formatted_sentence:
            sentence = f'{pre} {target}.{sense_label} {post}'
            saved_sentences.append(sentence)

    with open(f'{output_path}/{target}_{num_rows}.txt', 'w+') as fout:
        print(f'=================== {target.capitalize()} ===================\n', file=fout)
        print(f'{len(stats)} senses from {num_rows} rows', file=fout)
        print(f'From {dataset_name}', file=fout)

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

    with open(f'{output_path}/sentences/{target}_{num_rows}.dat', 'wb') as fout:
        pickle.dump(saved_sentences, fout)

    with open(logging_file, 'a') as flog:
        print('====================================\n', file=flog)

# %%


