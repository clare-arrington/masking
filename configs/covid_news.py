#%%
from base_wsi import get_data, make_predictions, make_clusters, sense_wrapper
import pandas as pd

def run_slice_predictions(input_path, output_path, corpus_name, slice_num, resume):
    data_path = f'{input_path}/{corpus_name}/slice_{slice_num}'
    sentence_path = f'{data_path}/target_sentences.pkl'
    target_path = f'{data_path}/target_information.pkl'
    masking_path = f'{output_path}/{corpus_name}/slice_{slice_num}'

    dataset_desc = f'{corpus_name.capitalize()} News Corpus'
    target_data = get_data(target_path, occurence_limit=25)
    targets = [[target] for target in target_data.target.unique() if target != 'coronavirus']
    print(f'{len(targets)} targets for masking')

    make_predictions(target_data, targets.copy(), dataset_desc, masking_path, 
                    subset_num=10000, resume_predicting=resume)

    make_clusters(target_data, dataset_desc, masking_path, 
                targets, resume_clustering=resume)
    sense_wrapper(sentence_path, masking_path)

def run_all_predictions(input_path, output_path, corpus_name, resume):
    data_path = f'{input_path}/{corpus_name}'
    target_path = f'{data_path}/target_information.pkl'
    sentence_path = f'{data_path}/target_sentences.pkl'
    masking_path = f'{output_path}/{corpus_name}'
    dataset_desc = f'{corpus_name.capitalize()} News Corpus'

    target_data = get_data(target_path, occurence_limit=25)
    target_data.set_index('word_index', inplace=True)

    targets = [[target] for target in target_data.target.unique() if target != 'coronavirus']
    print(f'{len(targets)} targets for masking')

    make_predictions(target_data, targets.copy(), dataset_desc, masking_path, 
                    subset_num=10000, resume_predicting=resume)

    make_clusters(target_data, targets, dataset_desc, masking_path, 
                resume_clustering=resume)

    sense_wrapper(sentence_path, masking_path)

## Pull data
input_path = '/data/arrinj/corpus_data/news/subset'
output_path = '/data/arrinj/masking_results/news'

for corpus_name in ['conspiracy', 'mainstream']: 
    run_all_predictions(input_path, output_path, corpus_name, resume=False)
    # for slice_num in range(0, 6):
        # run_slice_predictions(input_path, output_path, corpus_name, slice_num, resume=False)
        # print(f'{corpus_name.capitalize()} - slice {slice_num} done!\n\n')

# corpus_name = 'alternative'
# slice_num = 0

# %%
