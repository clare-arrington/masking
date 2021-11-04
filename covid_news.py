#%%
from base_wsi import get_data, make_predictions, cluster_predictions

## Pull data
corpus_name = 'unreliable'
data_path = '/data/arrinj/corpus_data/covid/subset'
sentence_path = f'{data_path}/target_sentences.csv'
target_path = f'{data_path}/target_information.csv'
output_path = f'/data/arrinj/masking_results/covid/{corpus_name}'

if 'reliable' in corpus_name:
    dataset_desc = f'{corpus_name.capitalize()} COVID Corpus'
    corpus_name = f'nela_{corpus_name}'
else:
    dataset_desc = 'Both reliable and unreliable COVID Corpora'
    corpus_name = None

#%%
target_data = get_data(sentence_path, target_path, corpus_name)

targets = [
    'corona', 'virus', 'covid', 'coronavirus', 'case', 'pandemic', 'crisis' 
    'mask', 'lockdown', 'quarantine', 'normal', 'death', 'trump']
targets = [[t] for t in targets]

#%%
make_predictions(target_data, dataset_desc, output_path, targets,
subset_num=5000, resume_predicting=False)

# cluster_predictions(dataset_desc, target_data, output_path, targets)

print('Done!')
# %%
