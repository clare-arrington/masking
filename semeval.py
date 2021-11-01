#%%
from base_wsi import get_data, make_predictions, perform_clustering, create_sense_sentences 

## Pull data
corpus_name = 'ccoha2'
# corpus_name = None
sentence_path = '/home/clare/Data/corpus_data/semeval/subset/target_sentences.csv'
target_path = '/home/clare/Data/corpus_data/semeval/subset/target_information.csv'
output_path = f'/home/clare/Data/masking_results/semeval/{corpus_name}'

if corpus_name == 'ccoha1':
    dataset_desc = 'SemEval 2020 Task \nCCOHA 1: 1810 - 1860'
elif corpus_name == 'ccoha2':
    dataset_desc = 'SemEval 2020 Task \nCCOHA 2: 1960 - 2010'
else:
    dataset_desc = 'SemEval 2020 Task \nCCOHA 1 and 2'
    corpus_name = None

#%%
target_data = get_data(sentence_path, target_path, 
    corpus_name, limit_occurences=False)

with open('/home/clare/Data/corpus_data/semeval/targets.txt') as fin:
    targets = fin.read().split()
    targets = [target.split('_')[0] for target in targets]

# targets = [[target] for target in target_data.target.unique()]

# targets = [
#     'bit', 'face', 'gas', 'head', 'lane',
#     'part', 'plane', 'record', 'word'] 
targets = [[target] for target in targets]

#%%
# make_predictions(
#     target_data, dataset_desc, output_path, 
#     targets, subset_num=5000)

# perform_clustering(
#     target_data, targets, dataset_desc, 
#     output_path)

print('Done!')
# %%
create_sense_sentences(sentence_path, target_path, output_path)

# %%
