#%%
from base_wsi import get_data, make_predictions, make_clusters
from sentence_maker import create_sense_sentences

## Pull data
corpus_name = '1800s'
input_path = '/home/clare/Data/corpus_data/semeval/subset_ai'
sentence_path = f'{input_path}/target_sentences.pkl'
target_path = f'{input_path}/target_information.pkl'
output_path = f'/home/clare/Data/masking_results/semeval/{corpus_name}_ai'

if corpus_name == '1800s':
    dataset_desc = 'SemEval 2020 Task \nCCOHA 1: 1810 - 1860'
elif corpus_name == '2000s':
    dataset_desc = 'SemEval 2020 Task \nCCOHA 2: 1960 - 2010'
else:
    dataset_desc = 'SemEval 2020 Task \n1800s (ccoha1) and 2000s (ccoha2)'
    corpus_name = None

#%%
target_data = get_data(target_path, base_count=50, corpus_name=corpus_name)
targets = [[target] for target in target_data.target.unique()]

# targets = [['ball']]

# with open('/home/clare/Data/corpus_data/semeval/targets.txt') as fin:
#     targets = fin.read().split()
#     targets = [[target.split('_')[0]] for target in targets]

#%%
make_predictions(
    target_data, targets.copy(), 
    dataset_desc, output_path, subset_num=15000)

make_clusters(
    target_data, targets, 
    dataset_desc, output_path, plot_clusters=True)

print('Done!')
# %%
create_sense_sentences(sentence_path, output_path)

# %%
