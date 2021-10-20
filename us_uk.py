#%%
import pandas as pd
from base_wsi import main

## Pull data
corpus_name = None
sentence_path = '/home/clare/Data/corpus_data/us_uk/subset/target_sentences.csv'
target_path = '/home/clare/Data/corpus_data/us_uk/subset/target_information.csv'

if corpus_name == 'bnc':
    dataset_desc = 'UK Corpus'
elif corpus_name == 'coca':
    dataset_desc = 'US Corpus'
else:
    dataset_desc = 'Both US and UK Corpora'

sentence_data = pd.read_csv(sentence_path, index_col='sent_id', usecols=['corpus', 'sent_id'])
if corpus_name is not None:
    sentence_data = sentence_data[sentence_data.corpus == corpus_name]
ids = sentence_data.index
print(f'{len(ids)} sentences pulled')

## TODO: Change this to load with index set
target_data = pd.read_csv(target_path)
target_data = target_data[target_data.sent_id.isin(ids)]
target_data.formatted_sentence = target_data.formatted_sentence.apply(eval)
print(f'{len(target_data)} target instances pulled')

#%%
## Set training info 
if corpus_name is None:
    corpus_name = 'all'

output_path = f'/home/clare/Data/masking_results/us_uk/{corpus_name}'
logging_file = f'{output_path}/targets.log'

target_counts = target_data.target.value_counts()
filtered_targets = target_counts[target_counts >= 20]
filtered_targets = list(filtered_targets.keys())
exclude = ['football', 'gas', 'hood', 'nappy', 'pavement',
           'rubber', 'subway', 'suspenders', 'sweets', 'vest']

## Get dissimilar
with open('/home/clare/Data/corpus_data/us_uk/truth/dissimilar.txt') as fin:
    dis_targets = fin.read().split()

targets = [[t] for t in dis_targets if t in filtered_targets and t not in exclude]

## Get similar
with open('/home/clare/Data/corpus_data/us_uk/truth/similar.txt') as fin:
    sim = fin.read().strip()

for pair in sim.split('\n'):
    uk_word, us_word = pair.split()
    if (uk_word in filtered_targets) and (us_word in filtered_targets):
        targets.append([us_word, uk_word])

#%%
main(target_data, dataset_desc, output_path, 
    logging_file, targets)

print('Done!')
# %%
