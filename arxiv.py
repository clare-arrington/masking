#%%
from base_wsi import filter_target_data, make_predictions, make_clusters
from sentence_maker import create_sense_sentences

## Pull data
corpus_name = 'ai'
input_path = '/home/clare/Data/corpus_data/arxiv/subset'
sentence_path = f'{input_path}/target_sentences.pkl'
target_path = f'{input_path}/target_information.pkl'
output_path = f'/home/clare/Data/masking_results/arxiv/{corpus_name}'

if corpus_name == 'ai':
    dataset_desc = 'ArXiv - Artificial Intelligence'
elif corpus_name == 'phys':
    dataset_desc = 'ArXiv - Classical Physics'
else:
    dataset_desc = 'ArXiv - AI and Physics'
    corpus_name = None

target_data = filter_target_data(target_path, base_count=50, corpus_name=corpus_name)
targets = [[target] for target in target_data.target.unique()]

#%%
make_predictions(
    target_data, targets.copy(), 
    dataset_desc, output_path, subset_num=15000)

make_clusters(
    target_data, targets, 
    dataset_desc, output_path, plot_clusters=True)

create_sense_sentences(sentence_path, output_path)

print('Done!')
# %%