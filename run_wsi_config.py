#%%
from base_wsi import get_data, make_predictions, make_clusters
from sentence_maker import create_sense_sentences
import json

## Set these
dataset_name = 'arxiv'
corpus_name = 'ai'
data_path = "/home/clare/Data"

## Pull data
with open(f"wsi_configs/{dataset_name}.json", "r") as read_file:
    config = json.load(read_file)

    input_path = f"{data_path}/corpus_data/{config['dataset_name']}/subset_ai"
    output_path = f"{data_path}/masking_results/semeval/{config['corpus_name']}"

    sentence_path = f"{input_path}/target_sentences.pkl"
    target_path = f"{input_path}/target_information.pkl"

    target_data = get_data(target_path, base_count=config['base_count'], corpus_name=corpus_name)
    targets = [[target] for target in target_data.target.unique()]

    dataset_desc = config['corpora_desc'][corpus_name]

    make_predictions(
        target_data, targets.copy(), 
        dataset_desc, output_path, subset_num=config['subset_num'])

    make_clusters(
        target_data, targets, 
        dataset_desc, output_path, plot_clusters=True)

    create_sense_sentences(sentence_path, output_path)

    print("Done!")
# %%
