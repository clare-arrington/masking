#%%
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import glob

dataset = 'coha'

main_path = f'/data/arrinj/masking_results/{dataset}/all/'
path = 'target_sense_labels.pkl'
cluster_data = pd.read_pickle(main_path + path)

sent_path = '_sense_sentences.pkl'
corpora_paths = glob.glob(
  main_path+'/*'+sent_path)

print('Reading in...')
all_sents = []
corpora = []
for corpus_path in sorted(corpora_paths):
  corpus = corpus_path[
    len(main_path):-len(sent_path)]
  corpora.append(corpus)
  print(f'\t{corpus} data')
  sents = pd.read_pickle(corpus_path)
  sents['corpus'] = corpus
  all_sents.append(sents)

sents = pd.concat(all_sents)
data = cluster_data.merge(sents, on='sent_idx', how='outer')
data = data.astype({
        "cluster": str, 
        "sent_idx": str})

#%%
colors = [
  '#f4a259', '#6d2e46', '#1a659e', 
  '#bc4749', '#6a994e', '#f4e285']
targets = data.target.unique()

for target in tqdm(targets):
  subset = data[data.target == target]

  if '0' in subset.cluster.unique():
    fig = px.histogram(
      subset, x="corpus", color="cluster",
      category_orders={
        "corpus":corpora},
      color_discrete_sequence=colors,
      title=f'Clusters for {target}')
    # fig.show()
    fig.write_image(main_path + f'plots/{target}.jpeg')

#%%