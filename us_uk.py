#%%
import pandas as pd
from base_wsi import main

def temp(wordset, output_path):
    if wordset == 'similar':
        dataset_name = 'US Corpus - Similar\n(Words have a UK equivalent with similar meaning, i.e. gas = petrol)'
    else:
        dataset_name = 'US Corpus - Dissimilar\n(Words have a UK equivalent with different meaning, i.e. football)'

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

    logging_file = f'{output_path}/{wordset}.log'

    return dataset_name, targets, logging_file

## Pull data
corpus = 'bnc'
input_path = f'/home/clare/Data/corpus_data/us_uk/subset/{corpus}_target_sents.csv'
data = pd.read_csv(input_path)
data.formatted_sentence = data.formatted_sentence.apply(eval)

#%%
## Set files for writing output 
output_path = f'/home/clare/Data/masking_results/us_uk/{corpus}'
logging_file = f'{output_path}/targets.log'

dataset_name = f'{corpus.upper()} Corpus'
targets = list(data.target.unique())

#%%
main(data, dataset_name, output_path, logging_file,
    targets, subset_num=5000)

print('Done!')
# %%
