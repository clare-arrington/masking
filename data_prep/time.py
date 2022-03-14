#%%
from get_data import pull_target_data, save_data

export_path  = '/home/clare/Data/corpus_data/time/ai_subset'
corpora_path = '/home/clare/Data/corpus_data/time/corpora'

with open(f'/home/clare/Data/corpus_data/time/targets.txt') as f:
    targets = f.read().split()

print(f'{len(targets)} targets loaded, ex. {", ".join(targets[:3])}')

corpus_targets = { 
    # 'coca' : targets
    # '1800s' : targets,
    # '2000s' : targets
    'ai'    : targets
}

#%%
sentence_data, target_data = \
    pull_target_data(corpus_targets, corpora_path, export_path)

#%%
save_data(sentence_data, target_data, export_path)

print('All done!')
# %%
