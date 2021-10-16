from get_data import pull_target_data, save_data

import pandas as pd
import random

path = '/home/clare/Data/corpus_data/arxiv-ai-phys'

def get_words(data, category, num_words, num_samples):
    words = list(data.nlargest(num_words, [category]).word)
    if num_samples is not None:
        return random.sample(words, num_samples)
    else:
        return words

def get_targets(data1, data2, num_words, num_samples=None):
    global_words  = get_words(data1, 'global', num_words, num_samples) 
    global_words += get_words(data2, 'global', num_words, num_samples)

    s4_words  = get_words(data1, 'self', num_words, num_samples)
    s4_words += get_words(data2, 'self', num_words, num_samples)

    targets = set(global_words + s4_words)
    return targets

data1 = pd.read_csv(f'{path}/cs.AI-physics.class-ph.csv')
data2 = pd.read_csv(f'{path}/physics.class-ph-cs.AI.csv')
targets = get_targets(data1, data2, 25)

#%%
rm_words = [
    'constraints',
    'medium', 
    'los',
    'fig',
    'rough',
    'die',
    'units'
]
# Low on AI: charge
# Low on Phys: network, polynomial, edge

for word in rm_words:
    if word in targets:
        targets.remove(word)

with open(f'{path}/targets.txt', 'w') as f:
    f.write('\n'.join(targets))

#%%
dataset_name = 'phys'
target_sents, non_target_sents = pull_target_data(targets, f'{path}/sentences/{dataset_name}.txt')

save_data(target_sents, non_target_sents, f'{path}/subset/{dataset_name}')


# %%
