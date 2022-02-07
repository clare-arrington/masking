#%%
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

def get_targets(path, num_words, num_samples=None):
    data1 = pd.read_csv(f'{path}/cs.AI-physics.class-ph.csv')
    data2 = pd.read_csv(f'{path}/physics.class-ph-cs.AI.csv')

    global_words  = get_words(data1, 'global', num_words, num_samples) 
    global_words += get_words(data2, 'global', num_words, num_samples)

    s4_words  = get_words(data1, 'self', num_words, num_samples)
    s4_words += get_words(data2, 'self', num_words, num_samples)

    targets = set(global_words + s4_words)
    return targets

def make_target_list(path):
    targets = get_targets(path, 25)

    rm_words = [
        'constraints',
        'medium', 
        'los',
        'fig',
        'rough',
        'die',
        'units'
    ]
    # Too low on AI: charge
    # Too low on Phys: network, polynomial, edge

    for word in rm_words:
        if word in targets:
            targets.remove(word)

    with open(f'{path}/targets.txt', 'w') as f:
        f.write('\n'.join(targets))

main_path = '/home/clare/Data/corpus_data/arxiv'

targets =  ['virus', 'bit', 'memory', 'long', 
            'float', 'web', 'worm', 'bug', 'structure',
            'cloud', 'ram', 'apple', 'cookie', 
            'spam',  'intelligence', 'artificial', 
            'time', 'work', 'action', 'goal', 'branch',
            'power', 'result', 'complex', 'root',
            'process', 'child', 'language', 'term',
            'rule', 'law', 'accuracy', 'mean', 
            'scale', 'variable', 'rest', 
            'normal', 'network', 'frame', 'constraint', 
            'subject', 'order', 'set', 'learn', 'machine',
            'problem', 'scale', 'large', 
            'model', 'based', 'theory', 'example', 
            'function', 'field', 'space', 'state', 
            'environment', 'compatible', 'case', 'natural', 
            'agent', 'utility', 'absolute', 'value', 
            'range', 'knowledge', 'symbol', 'true', 
            'class', 'object', 'fuzzy', 'global', 'local', 
            'search', 'traditional', 'noise', 'system']

print(f'{len(targets)} targets loaded, ex. {", ".join(targets[:3])}')

corpus_targets = {
    'ai' : targets}

corpora_path = f'{main_path}/sentences'
subset_path = f'{main_path}/subset'
pattern=r'[a-z]+'

#%%
sentence_data, target_data = pull_target_data(
    corpus_targets, corpora_path, subset_path, pattern)

save_data(sentence_data, target_data, subset_path)

print('All done!')
#%%
