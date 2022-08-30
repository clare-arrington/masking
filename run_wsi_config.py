#%%
from process_data import filter_target_data
from base_wsi import make_predictions, make_clusters
from sentence_maker import create_sense_sentences
from dotenv import dotenv_values
import json

## Set these
dataset_name = 'semeval'
run_every_corpora = True
combine_corpora = False
selected_corpus = None
embed=False
generate_new_predictions = True

##
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

    og_targets = set(og_targets)

    return 'together', config['dataset_desc'], target_paths, og_targets

## Pull data
data_path = dotenv_values(".env")['data_path']
input_path = f"{data_path}/corpus_data/{dataset_name}"
output_path = f"{data_path}/masking_results/{dataset_name}/"
with open(f"wsi_configs/{dataset_name}.json", "r") as read_file:
    config = json.load(read_file)

#%%
## Get corpus info based on settings
if run_every_corpora:
    corpora = config['corpora_data']
    corpora_info = [prep_corpus_info(input_path, config, corpus) for corpus in corpora]
elif combine_corpora:
    corpora_info = prep_info_together(input_path, config)
elif selected_corpus:
    corpora_info = prep_corpus_info(input_path, config, selected_corpus)
else:
    print('Nothing set for WSI')
    exit()

for corpus_info in corpora_info:
    print('Loading data')
    corpus_name, dataset_desc, target_paths, og_targets = corpus_info
    output_path = f"{data_path}/masking_results/{dataset_name}/{corpus_name}"

    # TODO: get alt targets for US / UK
    # TODO: when I drop extras, I should do so with respect to the different sections?
    target_data = filter_target_data(
        target_paths, og_targets, 
        config['min_sense_size'], 
        config['min_length'], 
        config['occurence_lim'], 
        config['subset_num'])

    if dataset_name == 'semeval':
        targets = [[t, t[:-3]] for t in og_targets]
    else:
        targets = target_data.target.unique()
        targets = [[t] for t in targets]
#%%
    ## TODO: predict on all instead of subset
    if generate_new_predictions:
        make_predictions(
            target_data.reset_index(), targets.copy(),
            dataset_desc, output_path, embed_sents=embed)
        print('Predicting done!')
    else:
        print('Skipping prediction step')
##%%
    ## TODO: cluster on subset, then match the rest
    make_clusters(
        target_data, targets, dataset_desc, 
        config['min_sense_size'],
        output_path, embed_sents=embed, 
        print_clusters=True)
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
        create_sense_sentences(sentence_path, output_path, corpus_name)

print("Done!")
# %%
