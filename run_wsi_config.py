#%%
from process_data import filter_target_data
from predict_main import make_predictions
from cluster_main import make_clusters
from sentence_maker import create_sense_sentences
from dotenv import dotenv_values
import json

### This is the main file for the masking portion

CLUSTER_OPTIONS = [
    'all_together', # merge sentences from all corpora, then do WSI
    'all_separate', # perform WSI for each corpus independently
    'single_corpus' # perform WSI for one corpus, designated by 'selected_corpus'
    ]

## Set these args
cluster_option = CLUSTER_OPTIONS[0]
generate_new_predictions = True
dataset_name = 'semeval'
selected_corpus = '2000s'
embed_sents = False # use BERT embeddings for clustering instead of MLM prediciton vectors

## Get information about corpus and set paths
def prep_corpus_info(input_path, config, corpus_name):
    dataset_desc, target_file = config['corpora_data'][corpus_name]
    target_path = f"{input_path}/targets/{target_file}"
    with open(target_path, 'r') as f:
        og_targets = f.read().split()

    subset_path = f"{input_path}/subset/{corpus_name}_indexed"
    target_path = f"{subset_path}_words.pkl"
    target_paths = {corpus_name : target_path}

    return corpus_name, dataset_desc, target_paths, og_targets

## TODO: double check this is together
def prep_info_together(input_path, config):
    target_paths = {}
    og_targets = []
    for corpus, (_, target_file) in config['corpora_data'].items():    
        target_path = f"{input_path}/targets/{target_file}"
        with open(target_path, 'r') as f:
            og_targets.extend(f.read().split())

        subset_path = f"{input_path}/subset/{corpus}_indexed"
        target_path = f"{subset_path}_words.pkl"
        target_paths[corpus] = target_path

    return 'together', config['dataset_desc'], target_paths, set(og_targets)

## Pull data
data_path = dotenv_values(".env")['data_path']
input_path = f"{data_path}/corpus_data/{dataset_name}"
output_path = f"{data_path}/masking_results/{dataset_name}"
with open(f"configs/{dataset_name}.json", "r") as read_file:
    config = json.load(read_file)

#%%
## Get corpus info based on WSI setting (grouped, solo, etc)
if cluster_option == 'all_together':
    corpora_info = [prep_info_together(input_path, config)]
elif cluster_option == 'all_separate':
    corpora = config['corpora_data']
    corpora_info = [prep_corpus_info(input_path, config, corpus) for corpus in corpora]
elif cluster_option == 'single_corpus' and selected_corpus in config['corpora_data']:
    corpora_info = [prep_corpus_info(input_path, config, selected_corpus)]
else:
    print('Nothing set for WSI')
    exit()

for corpus_info in corpora_info:
    print('Loading data')
    corpus_name, dataset_desc, target_paths, og_targets = corpus_info
    save_path = f"{output_path}/{corpus_name}"
    print(save_path)

    # TODO: get alt targets for US / UK
    # og_targets = [
    #     'face_nn','head_nn']
    target_data = filter_target_data(
        target_paths, og_targets, 
        config['min_sense_size'], 
        config['min_length'], 
        config['occurence_lim'])

    if dataset_name == 'semeval':
        ## original corpus is POS labeled; need to trim off for BERT
        targets = [[t, t[:-3]] for t in og_targets] 
    else:
        targets = target_data.target.unique()
        targets = [[t] for t in targets]
#%%
    if generate_new_predictions:
        make_predictions(
            target_data.reset_index(), targets.copy(),
            dataset_desc, save_path, embed_sents=embed_sents)
        print('Predicting done!')
    else:
        print('Skipping prediction step')
##%%
    make_clusters(
        target_data, targets, dataset_desc, 
        config['min_sense_size'],
        save_path, embed_sents=embed_sents, 
        print_clusters=True, plot_clusters=True)
    print('Clustering done!')
##%%
    del target_data
    ## If WSI was done together, we need to split the corpora up for the sentences
    if corpus_name == 'together':
        corpora = config['corpora_data']
    else: 
        corpora = [corpus_name]

    for corpus_name in corpora:
        subset_path = f"{input_path}/subset/{corpus_name}_indexed"
        sentence_path = f"{subset_path}_sentences.pkl"
        create_sense_sentences(sentence_path, save_path, corpus_name)

print("Done!")
# %%
