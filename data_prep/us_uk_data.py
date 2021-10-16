#%%
from get_data import pull_target_data, save_data

def get_us_uk_targets(path, get_us=False, get_uk=False):
    targets = []
    ## Get dissimilar
    with open(f'{path}/dissimilar.txt') as fin:
        dis = fin.read().split()
        targets.extend(dis)

    ## Get similar
    with open(f'{path}/similar.txt') as fin:
        sim = fin.read().strip()
        for pair in sim.split('\n'):
            uk_word, us_word = pair.split()
            if get_us:
                targets.append(us_word)
            elif get_uk:
                targets.append(uk_word)

    return targets

#%%
dataset_name = 'bnc'
corpora_path = '/home/clare/Data/corpus_data/us_uk'

if dataset_name == 'bnc':
    targets = get_us_uk_targets(f'{corpora_path}/truth', get_uk=True)
elif dataset_name == 'coca':
    targets = get_us_uk_targets(f'{corpora_path}/truth', get_us=True)

target_sents, non_target_sents = pull_target_data(targets, f'{corpora_path}/corpora/{dataset_name}.txt')
save_data(target_sents, non_target_sents, f'{corpora_path}/subset/{dataset_name}')
