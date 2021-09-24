#%%
import pandas as pd
from base_wsi import main

## Pull data
input_path = '/home/clare/Data/corpus_data/semeval/corpora/ccoha2_target_sents.csv'
data = pd.read_csv(input_path)
data.formatted_sentence = data.formatted_sentence.apply(eval)

targets = list(data.target.unique())
dataset_name = 'SemEval 2020 Task \nCCOHA 1960 - 2010'

#%%
## Set training info
train_type = 'preds' 

output_path = f'/home/clare/Data/masking_results/semeval/{train_type}'
logging_file = f'{output_path}/targets.log'

# load_sentence_path='/home/clare/Data/masking_results/semeval/preds/clusters'
load_sentence_path = ''

#%%
if train_type == 'preds':
    use_reps = False
elif train_type == 'subs':
    use_reps = True

main(data, dataset_name, output_path, logging_file,
    targets, load_sentence_path=load_sentence_path,
    use_representatives=use_reps)

print('Done!')
# %%