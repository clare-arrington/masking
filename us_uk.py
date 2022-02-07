#%%
from base_wsi import get_data, make_predictions, make_clusters
from sentence_maker import create_sense_sentences 

def prep_targets(target_data, corpus_name):

    if corpus_name is not None:
        return [[t] for t in target_data.target.unique()]

    else: ## TODO: this all is a mess 

        targets = target_data.target.unique()
    
        ## TODO: solve this problem 
        ## These are removed from the dissimilar group b/c they're also in the similar group
        exclude = ['football', 'gas', 'hood', 'nappy', 'pavement',
                'rubber', 'subway', 'suspenders', 'sweets', 'vest']

        ## Get dissimilar
        with open('/data/arrinj/corpus_data/us_uk/truth/dissimilar.txt') as fin:
            dis_targets = fin.read().split()

        targets = [[t] for t in dis_targets if t in targets and t not in exclude]

        ## Get similar
        with open('/data/arrinj/corpus_data/us_uk/truth/similar.txt') as fin:
            sim = fin.read().strip()

        ## TODO: the idea here is that we would have both terms grouped together and do WSI on the combined group
        for pair in sim.split('\n'):
            uk_word, us_word = pair.split()
            if corpus_name == 'all' and (uk_word in filtered_targets) and (us_word in filtered_targets):
                targets.append([us_word, uk_word])
            elif corpus_name == 'bnc' and (uk_word in filtered_targets):
                targets.append([uk_word])
            elif corpus_name == 'coca' and (us_word in filtered_targets):
                targets.append([us_word])
            
        return targets

## Pull data
corpus_name = 'coca'
sentence_path = '/data/arrinj/corpus_data/us_uk/subset/target_sentences.pkl'
target_path = '/data/arrinj/corpus_data/us_uk/subset/target_information.pkl'
output_path = f'/data/arrinj/masking_results/us_uk/{corpus_name}'

if corpus_name == 'bnc':
    dataset_desc = 'UK Corpus'
elif corpus_name == 'coca':
    dataset_desc = 'US Corpus'
else:
    dataset_desc = 'Both US and UK Corpora'
    corpus_name = None

#%%
target_data = get_data(target_path, base_count=100, corpus_name=corpus_name)
targets = prep_targets(target_data, corpus_name)

#%%
make_predictions(target_data, targets.copy(), dataset_desc, output_path, subset_num=15000)

make_clusters(
    target_data, targets, 
    dataset_desc, output_path)

# %%
create_sense_sentences(sentence_path, output_path)

print('Done!')

# %%
