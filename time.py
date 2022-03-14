#%%
from base_wsi import filter_target_data, pull_rows, make_predictions, make_clusters
from sentence_maker import create_sense_sentences

## Pull data
corpus_name = 'ai'
input_path = '/home/clare/Data/corpus_data/time/ai_subset'

sentence_path = f'{input_path}/target_sentences.pkl'
target_path = f'{input_path}/target_information.pkl'
output_path = f'/home/clare/Data/masking_results/time/{corpus_name}'

if corpus_name == '1800s':
    dataset_desc = 'SemEval 2020 Task \nCCOHA 1: 1810 - 1860'
elif corpus_name == '2000s':
    dataset_desc = 'SemEval 2020 Task \nCCOHA 2: 1960 - 2010'
elif corpus_name == 'coca':
    dataset_desc = 'Corpus of Contemporary American English'
elif corpus_name == 'ai':
    dataset_desc = 'ArXiv AI'

#%%
remove = ['time', 'work', 'like', 'long']
target_data = filter_target_data(target_path, base_count=100, corpus_name=corpus_name)
targets = [[target] for target in target_data.target.unique() if target not in remove]

## Filter out non-wanted targets
target_data = target_data[~target_data['target'].isin(remove)]
print(len(target_data), ' target rows after removing some targets')

#%%
target_rows = pull_rows(target_data.reset_index(), subset_num=7500)

# for k, v in target_rows.items():
#     if len(v) > 5000:
#         print(k, len(v))

#%%
make_predictions(
    target_rows, targets.copy(), 
    dataset_desc, output_path, resume_predicting=False)

make_clusters(
    target_data, targets, 
    dataset_desc, output_path, plot_clusters=False)

# %%
create_sense_sentences(sentence_path, output_path)

print('Done!')
# %%
