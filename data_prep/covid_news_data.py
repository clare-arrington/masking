#%%
from get_data import pull_target_data, save_data

#%%
main_path = '/data/arrinj/corpus_data/covid'

targets = ['corona', 'virus', 'covid', 'coronavirus', 'case', 'pandemic', 'crisis' 
           'mask', 'lockdown', 'quarantine', 'normal', 'death', 'trump']

corpus_targets = { 
    'nela_reliable': targets,
    'nela_unreliable' : targets
}

# corpora_path = f'{main_path}/corpora'
corpora_path = main_path
subset_path = f'{main_path}/subset'
pattern=r'[a-z]+'

#%%
sentence_data, target_data = \
    pull_target_data(corpus_targets, corpora_path, subset_path, pattern)

#%%
save_data(sentence_data, target_data, subset_path)

print('All done!')
# %%
