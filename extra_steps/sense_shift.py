#%%
import pandas as pd

sentence_path = '/home/clare/Data/corpus_data/semeval/subset/target_sentences.csv'
sentence_data = pd.read_csv(sentence_path)
sentence_data.set_index('sent_id', inplace=True) 

target_path = '/home/clare/Data/masking_results/semeval/all_1/target_sense_labels.csv'
target_data = pd.read_csv(target_path)
target_data.set_index('word_index', inplace=True) 

label_path = '/home/clare/Data/corpus_data/semeval/truth/binary.txt'
labels = {'Shifted':[], 'Unshifted':[]  }
with open(label_path) as fin:
    og_targets = fin.read().strip().split('\n')
    for target in og_targets:
        word, label = target.split('\t')
        word, pos = word.split('_')
        if label == '1':    
            labels['Shifted'].append(word)
        else:
            labels['Unshifted'].append(word)

#%%
corpus_length = {}
corpora = ['ccoha1', 'ccoha2']
for corpus in corpora:
    s = sentence_data[sentence_data.corpus == corpus]
    c_targets = target_data[target_data.sent_id.isin(s.index)]
    corpus_length[corpus] = len(c_targets)

    print(f'Corpus {corpus.upper()} : {len(c_targets)} targets')

#%%

missed = ['attack', 'bit', 'circle', 'edge', 'head', 'land',
          'lass', 'rag', 'stab', 'thump', 'tip']

for label, targets in labels.items():
    print(f'\n\n=====================================')
    print(f'========= {label} Targets ===========')
    print(f'=====================================')
    for target in targets:
        if target in missed:
            title = f'{target.capitalize()} *'
        else:
            title = f'{target.capitalize()}'

        print(f'\n=========== {title} ===========')
        rows = target_data[target_data.target == target]
        print(f'{len(rows)} target occurences')

        sent_data = sentence_data.loc[rows.sent_id]

        for cluster in rows.cluster.unique():
            ids = rows[rows.cluster == cluster].sent_id
            sents = sentence_data.loc[ids]
            print(f'\n\t== Cluster {cluster} ==')
            print(f'\t{len(ids)} occurences\n')

            for corpus in corpora:
                subset = sents[sents.corpus == corpus]
                prop = len(subset) / len(sents)
                print(f'{prop:.2f} of cluster from {corpus}')

            print()
            for corpus in corpora:
                subset = sents[sents.corpus == corpus]
                all_corpus = sent_data[sent_data.corpus == corpus]
                prop = len(subset) / len(all_corpus)
                print(f'{prop:.2f} of {corpus} in cluster')

            # for corpus in corpora:
            #     prop = len(subset) / corpus_length[corpus]
            #     print(f'\t{prop:.3f} of targets in {corpus} in this cluster')

# %%
