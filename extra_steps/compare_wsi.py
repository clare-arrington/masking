#%%
from dotenv import dotenv_values
import pandas as pd

dataset = 'coha'
target_file = 'gems_targets.txt'

data_path = dotenv_values("../.env")['data_path']
masking_path = f'{data_path}/masking_results/{dataset}'
target_path = f'{data_path}/corpus_data/{dataset}/targets/'
with open(target_path+target_file) as fin:
    targets = fin.read().split()

#%%
cluster_results = []
corpus_names = [str(n) for n in range(1910,2010,10)]
for corpus in corpus_names:
    target_senses = {target:0 for target in targets}
    # target_counts = {}
    full_path = f'{masking_path}/{corpus}/clustering.log'

    with open(full_path) as fout:
        num_senses = 0
        target = None
        for line in fout.read().split('\n'):
            if 'rows' in line and ':' in line:
                target, num_rows = line.lower().split(' : ')
                num_rows = num_rows[:-5]
                num_senses = 0
                # target_counts[target] = num_rows
            elif ':' in line and 'time' not in line:
                num_senses += 1
            elif target and '===' in line:
                target_senses[target] = num_senses

        # cluster_results.extend([target_senses, target_counts])
        cluster_results.append(target_senses)
        
df = pd.DataFrame(cluster_results, index=corpus_names, dtype=int).T
#%%
df.to_csv(f'{masking_path}/wsi_summary.csv')
# %%
