#%%
from get_data import pull_target_data, save_data

def get_us_uk_targets(path, get_us=False, get_uk=False):
    targets = set()
    ## Get dissimilar
    with open(f'{path}/dissimilar.txt') as fin:
        dis = fin.read().split()
        targets.update(dis)

    ## Get similar
    with open(f'{path}/similar.txt') as fin:
        sim = fin.read().strip()
        for pair in sim.split('\n'):
            uk_word, us_word = pair.split()
            if get_us:
                targets.add(us_word)
            elif get_uk:
                targets.add(uk_word)

    ## Get spelling
    with open(f'{path}/spelling.txt') as fin:
        spell = fin.read().strip()
        for pair in spell.split('\n'):
            uk_word, us_word = pair.split()
            if get_us:
                targets.add(us_word)
            elif get_uk:
                targets.add(uk_word)

    return list(targets)

#%%
main_path = '/data/arrinj/corpus_data/us_uk'

corpus_targets = { 
    'bnc': get_us_uk_targets(f'{main_path}/truth', get_uk=True),
    'coca' : get_us_uk_targets(f'{main_path}/truth', get_us=True)
}

corpora_path = f'{main_path}/corpora'
subset_path = f'{main_path}/subset'

#%%
sentence_data, target_data = \
    pull_target_data(corpus_targets, corpora_path, subset_path)

#%%
save_data(sentence_data, target_data, subset_path)

print('All done!')
# %%
