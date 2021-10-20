#%%
from os import remove
from get_data import pull_target_data, save_data

def get_targets(main_path, remove_pos=False):
    with open(f'{main_path}/truth/binary.txt') as fin:
        og_targets = fin.read().strip().split('\n')
    
    targets = []
    for target in og_targets:
        word, label = target.split('\t')
        if remove_pos:
            word, pos = word.split('_')
        targets.append(word)

    return targets

# print(f'{len(targets)} targets loaded, ex. {targets[0]}')

main_path = '/home/clare/Data/corpus_data/semeval'
targets = get_targets(main_path)

corpus_targets = {
    'ccoha1' : targets, 
    'ccoha2' : targets}

corpora_path = f'{main_path}/corpora'
subset_path = f'{main_path}/subset'
pattern=r'[a-z]+_[a-z]{2}|[a-z]+'

#%%
sentence_data, target_data = \
    pull_target_data(corpus_targets, corpora_path, subset_path, pattern)
save_data(sentence_data, target_data, subset_path)

print('All done!')
#%%
