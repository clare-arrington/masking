#%%
from base_wsi import get_data, make_predictions, make_clusters, create_sense_sentences

def run_slice_predictions(input_path, output_path, corpus_name, slice_num, resume):
    data_path = f'{input_path}/{corpus_name}/slice_{slice_num}'
    sentence_path = f'{data_path}/target_sentences.pkl'
    target_path = f'{data_path}/target_information.pkl'
    masking_path = f'{output_path}/{corpus_name}/slice_{slice_num}'

    dataset_desc = f'{corpus_name.capitalize()} News Corpus'
    target_data = get_data(target_path, occurence_limit=25)
    targets = [[target] for target in target_data.target.unique() if target != 'coronavirus']
    print(f'{len(targets)} targets for masking')

    make_predictions(target_data, dataset_desc, masking_path, targets.copy(),
                    subset_num=10000, resume_predicting=resume)

    make_clusters(target_data, dataset_desc, masking_path, 
                targets, resume_clustering=resume)
    create_sense_sentences(sentence_path, masking_path)

## Pull data
input_path = '/data/arrinj/corpus_data/news/subset'
output_path = '/data/arrinj/masking_results/news'

for corpus_name in ['mainstream']: # 'alternative'
    for slice_num in range(0, 6):
        run_slice_predictions(input_path, output_path, corpus_name, slice_num, resume=False)
        print(f'{corpus_name.capitalize()} - slice {slice_num} done!\n\n')

# corpus_name = 'mainstream'
# slice_num = 0

# %%
