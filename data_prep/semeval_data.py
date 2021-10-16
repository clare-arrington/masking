#%%
from get_data import pull_target_data, save_data

corpus_names = ['ccoha1', 'ccoha2']
main_path = '/home/clare/Data/corpus_data/semeval'

with open(f'{main_path}/truth/binary.txt') as fin:
    og_targets = fin.read().strip().split('\n')
    targets = []
    for target in og_targets:
        word, label = target.split('\t')
        # word, pos = word.split('_')
        targets.append(word)

print(f'{len(targets)} targets loaded, ex. {targets[0]}')

corpora_path = f'{main_path}/corpora'
subset_path = f'{main_path}/subset'
pattern=r'[a-z]+_[a-z]{2}|[a-z]+'

#%%
sentence_data, target_data = \
    pull_target_data(targets, corpora_path, subset_path, corpus_names, pattern)
save_data(sentence_data, target_data, subset_path)

print('All done!')
#%%
