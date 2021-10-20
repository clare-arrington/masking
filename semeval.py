#%%
from base_wsi import main, create_sense_sentences
import pandas as pd

## Pull data
# corpus_name = 'ccoha1'
corpus_name = None
sentence_path = '/home/clare/Data/corpus_data/semeval/subset/target_sentences.csv'
target_path = '/home/clare/Data/corpus_data/semeval/subset/target_information.csv'

if corpus_name == 'ccoha1':
    dataset_desc = 'SemEval 2020 Task \nCCOHA 1: 1810 - 1860'
elif corpus_name == 'ccoha2':
    dataset_desc = 'SemEval 2020 Task \nCCOHA 2: 1960 - 2010'
else:
    dataset_desc = 'SemEval 2020 Task \nCCOHA 1 and 2'

sentence_data = pd.read_csv(sentence_path)
if corpus_name is not None:
    sentence_data = sentence_data[sentence_data.corpus == corpus_name]
ids = list(sentence_data.sent_id)
print(f'{len(ids)} sentences pulled')

target_data = pd.read_csv(target_path)
target_data = target_data[target_data.sent_id.isin(ids)]
target_data.formatted_sentence = target_data.formatted_sentence.apply(eval)
print(f'{len(target_data)} target instances pulled')

targets = list(target_data.target.unique())

#%%
## Set training info 
if corpus_name is None:
    corpus_name = 'all'

output_path = f'/home/clare/Data/masking_results/semeval/{corpus_name}'
logging_file = f'{output_path}/targets.log'

#%%
main(target_data, dataset_desc, output_path, 
    logging_file, targets)

print('Done!')
# %%
create_sense_sentences(sentence_data, output_path)
