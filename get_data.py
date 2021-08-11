#%%
import pickle
import pandas as pd
import re
import tqdm

def generate_us_uk_targets():
    targets = []
    ## Get dissimilar
    with open('../data/us_uk/dissimilar.txt') as fin:
        dis = fin.read().split()
        # targets.extend(sorted(random.sample(dis, 25)))
        targets.extend(dis)

    ## Get similar
    with open('../data/us_uk/similar.txt') as fin:
        sim = fin.read().strip()
        us = []
        for pair in sim.split('\n'):
            uk_word, us_word = pair.split()
            us.append(us_word)
        targets.extend(us)

    target_path = f'../data/us_uk/targets.txt'
    with open(target_path, 'w') as fout:
        for target in targets:
            print(target, file=fout)

target_path = f'../data/us_uk/targets.txt'
with open(target_path) as fin:
    targets = fin.read().split()

#%%
print('Parsing Sentences')
# word_indices = {word[:-3]:0 for word in targets}
word_indices = {word:0 for word in targets}
non_target_sents = []
sentence_data = []

# path = f'/home/rozek/ClareWork/data/semeval/corpora/ccoha1'
path = f'/home/rozek/ClareWork/data/us_uk/coca'
pattern = re.compile(r'[a-z]+')

with open(f'{path}.txt') as fin:
    for line in tqdm.tqdm(fin.readlines()):
        line = line.lower().strip()
        # words = line.split()
        words = re.findall(pattern, line)
        found_targets = set(targets).intersection(set(words))

        if found_targets == set():
            non_target_sents.append(line.lower())
            continue
        
        for target in found_targets:
            # target = target[:-3]
        
            index = word_indices[target]
            word_indices[target] += 1
            word_index = f'{target}.{index}'

            # If we want a window, add option here
            index = words.index(target)
            pre = ' '.join(words[:index])
            post = ' '.join(words[index + 1:])

            formatted_sent = (pre, target, post)
            length = len(pre) + len(post)

            info = [word_index, target, line, formatted_sent, length]
            sentence_data.append(info)

## Convert to dataframes
sentence_data = pd.DataFrame(sentence_data, columns=['word_index', 'target', 'sentence', 'formatted_sentence', 'length'])
print('\n\nTarget Counts')
print(sentence_data.target.value_counts())

# sentence_data.head()

# %%
with open(f'{path}_non_target.dat', 'wb') as pout:
    pickle.dump(non_target_sents, pout)

sentence_data.to_csv(f'{path}_target_sents.csv', index=False)
# %%
