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
main_path = '/data/arrinj/corpus_data/us_uk'

corpus_targets = { 
    'bnc': get_us_uk_targets(f'{main_path}/truth', get_uk=True),
    'coca' : get_us_uk_targets(f'{main_path}/truth', get_us=True)
}

corpora_path = f'{main_path}/corpora'
subset_path = f'{main_path}/subset'
pattern=r'[a-z]+'

#%%
sentence_data, target_data = \
    pull_target_data(corpus_targets, corpora_path, subset_path, pattern)

#%%
save_data(sentence_data, target_data, subset_path)

print('All done!')
# %%
