#%%
from get_data import preprocess_data, pull_from_preprocessed_data
import pandas as pd

def make_time_slices(data, corpus_name, path):
    date_ranges = [ ('2020-01', '2020-03'),
                    ('2020-04', '2020-06'),
                    ('2020-07', '2020-09'),
                    ('2020-10', '2020-12'),
                    ('2021-01', '2021-03'),
                    ('2021-04', '2021-07')
    ]

    for slice, date_range in enumerate(date_ranges):
        start, end = date_range
        rows = data[data.date.between(
            f'{start}-01', f'{end}-31')]
        print(start, len(rows))

        preprocess_data(list(rows.content), corpus_name, 
                    f'{path}/{corpus_name}_slice_{slice}.pkl')       

# main_path = '/data/arrinj/corpus_data/news'
# data = pd.read_pickle('/data/nela/corpus.pickle')
# data = data.drop(columns=['id', 'timestamp', 'title'])
# labels = pd.read_csv(f'{main_path}/labels.csv')
# labels.set_index('source', inplace=True)

# data = data.join(labels, on='source').dropna()
# data.cluster = data.cluster.astype(int)

# mainstream = data[data.cluster == 2]
# mainstream = mainstream.drop(
#     mainstream[mainstream.source.isin(['oann', 'foreignpolicy'])].index)
# alternative = data[data.cluster == 1]

# main_path += '/corpora/time_slices'

# make_time_slices(mainstream, 'mainstream', main_path)
# make_time_slices(alternative, 'alternative', main_path)

#%%
covid_targets = [ 'corona', 'virus', 'covid', 
            'coronavirus', 'case', 'pandemic',
            'crisis', 'mask', 'lockdown', 
            'quarantine', 'normal', 'death', 
            'vaccine', 'mask', 
            'distance', 'test', 'antiviral'
            'curve', 'epidemic', 'spread', 
            'positive', 'business', 'essential',
            'hero', 'work', 'drug', 'risk',
            'safe', 'immune', 'sick'
            ]

# social_distancing
# banned, testing, studies, needs
generic_targets = [
    'need', 'ban', 'safety', 
    'climate', 'wear', 'bill',
    'states', 'change',
    'food', 'report',
    'article', 'air', 'health',
    'public', 'cause', 'system',
    'research', 'vitamin', 'product',
    'symptom', 'study', 'healthy',
    'natural', 'wellness',
    'effect'
]

targets = covid_targets + generic_targets

main_path = '/data/arrinj/corpus_data/news'

for corpus_name in ['alternative', 'mainstream']:
    for slice in range(0, 6):
        data_path = f'{main_path}/corpora/time_slices/{corpus_name}_slice_{slice}.csv'
        save_path = f'{main_path}/subset/{corpus_name}/slice_{slice}'
        pull_from_preprocessed_data(data_path, save_path, targets)

print('All done!')
# %%
