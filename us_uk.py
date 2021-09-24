#%%
import pandas as pd
from base_wsi import main

## Pull data
input_path = '/home/clare/Data/corpus_data/us_uk/subset/coca_target_sents.csv'
data = pd.read_csv(input_path)
data.formatted_sentence = data.formatted_sentence.apply(eval)

#%%
## Set dataset info
wordset = 'similar'
if wordset == 'similar':
    dataset_name = 'US Corpus - Similar\n(Words have a UK equivalent with similar meaning, i.e. gas = petrol)'
else:
    dataset_name = 'US Corpus - Dissimilar\n(Words have a UK equivalent with different meaning, i.e. football)'

## Set files for writing output 
output_path = '/home/clare/Data/masking_results/us_uk/subs'
logging_file = f'{output_path}/{wordset}.log'

load_sentence_path='/home/clare/Data/masking_results/us_uk_15/preds/clusters'

## Get targets
target_path = f'/home/clare/Data/corpus_data/us_uk/truth/{wordset}.txt'
with open(target_path) as fin:
    if wordset == 'similar':
        targets = []
        pairs = fin.read().strip()
        for pair in pairs.split('\n'):
            uk_word, us_word = pair.split()
            targets.append(us_word)
    else:
        targets = fin.read().split()

#%%
main(data, dataset_name, output_path, logging_file,
    targets, load_sentence_path=load_sentence_path,
    use_representatives=True)

print('Done!')
# %%

import numpy as np
# TODO: this is out of the targets, is that okay?
for word, count in data.target.value_counts().iteritems():
    frac = count/len(data)
    prob = (np.sqrt(frac/0.001) + 1) * (0.001/frac)
    
    print(word, count, f'{prob:.2f} =>', int(count*prob))
# %%
