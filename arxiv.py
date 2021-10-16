#%%
import pandas as pd
from base_wsi import main

## Pull data
corpus = 'ai'
dataset_name = 'arxiv'
input_path = f'/home/clare/Data/corpus_data/{dataset_name}/subset/{corpus}_target_sents.csv'
data = pd.read_csv(input_path)
data.formatted_sentence = data.formatted_sentence.apply(eval)

targets = list(data.target.unique())
dataset_desc = f'{corpus.upper()} Corpus'

#%%
output_path = f'/home/clare/Data/masking_results/{dataset_name}/{corpus}'
logging_file = f'{output_path}/targets.log'

#load_sentence_path='/home/clare/Data/masking_results/semeval/preds/clusters'

#%%
main(data, dataset_desc, output_path, logging_file, targets)

print('Done!')
# %%