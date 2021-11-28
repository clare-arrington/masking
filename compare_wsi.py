#%%
import pandas as pd

data_path = '/data/arrinj'
masking_path = f'{data_path}/masking_results/news'
with open(f'{data_path}/corpus_data/news/targets.txt') as fin:
    targets = fin.read().split()

cluster_results = []
corpus_names = ["Alternative", "Mainstream"]
for corpus in corpus_names:
    for slice_num in range(0, 6):
        target_senses = {}
        target_counts = {}
        full_path = f'{masking_path}/{corpus.lower()}/slice_{slice_num}/clustering.log'

        with open(full_path) as fout:
            num_senses = 0
            target = None
            for line in fout.read().split('\n'):
                if 'rows' in line and ':' in line:
                    target, num_rows = line.split(' : ')
                    num_rows = num_rows[:-5]
                    num_senses = 0
                    target_counts[target] = num_rows
                elif ':' in line and 'time' not in line:
                    num_senses += 1
                elif target and '===' in line:
                    target_senses[target] = num_senses

            target_senses[target] = num_senses
        cluster_results.extend([target_senses, target_counts])

#%%
iterables = [ corpus_names, 
             [f"Slice {slice_num}" for slice_num in range(0,6)],
             ['Num. Senses', 'Word Freq']
             ]

cols = pd.MultiIndex.from_product(iterables, names=["corpus", "slice", "column values"])

df = pd.DataFrame(cluster_results).T
df.sort_index(inplace=True)
df.columns = cols

df.to_csv(f'{masking_path}/wsi_summary.csv')
# %%
