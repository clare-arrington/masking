from wsi.lm_bert import LMBert
from wsi.WSISettings import DEFAULT_PARAMS, WSISettings
from log import record_time
from typing import List
from pathlib import Path
import pandas as pd
from glob import glob
import pickle

## Main file for MLM prediction, called from run_wsi_config

def make_predictions(
    target_data: pd.DataFrame,
    targets: List[str],
    dataset_desc: str,
    output_path: str,
    resume_predicting=False,
    embed_sents=False
    ):

    settings = DEFAULT_PARAMS._asdict()
    settings = WSISettings(**settings)

    ## Load BERT model
    lm = LMBert(settings)

    if embed_sents:
        Path(f'{output_path}/vectors').mkdir(parents=True, exist_ok=True)
    else:
        Path(f'{output_path}/predictions').mkdir(parents=True, exist_ok=True)
    logging_file = f'{output_path}/prediction.log'

    ## Start the new logging file for this run
    if not resume_predicting:
        with open(logging_file, 'w') as flog:
            print(dataset_desc, file=flog)
            # print(f'\n{len(target_data):,} rows loaded', file=flog)
            print(f'{len(targets)} targets loaded\n', file=flog)
    else:
        already_predicted = glob(f'{output_path}/predictions/*.pkl')
        skip_targets = [path.split('/')[-1][:-4] for path in already_predicted]
        print(f'{len(skip_targets)} targets already predicted')

        remove_targets = []
        for target in targets:
            if target[0] in skip_targets:
                remove_targets.append(target)
        print(f'Removing {len(remove_targets)} targets')

        for target in remove_targets:
            targets.remove(target)
        print(f'{len(targets)} targets going to be clustered')

    for n, target_alts in enumerate(sorted(targets)):
        # break
        target = target_alts[0]
        print(f'\n{n+1} / {len(targets)} : {" ".join(target_alts)}')

        data_subset = target_data[target_data.target == target]
        num_rows = len(data_subset)

        with open(logging_file, 'a') as flog:
            print('====================================\n', file=flog)
            print(f'{target.capitalize()} : {num_rows} rows', file=flog)
            if len(target_alts) > 1:
                print(f'Alt form: {target_alts[1]}', file=flog)

            print(f'\tPredicting for {num_rows} rows...')
            print('\n' + record_time('start'), file=flog)
            if embed_sents:
                vectors = lm.embed_sents(data_subset, target_alts[-1])
                print(record_time('end') + '\n', file=flog)

                with open(f'{output_path}/vectors/{target}.pkl', 'wb') as vp:
                    pickle.dump(vectors, vp, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'\tVectors saved')

            else:
                predictions = lm.predict_sent_substitute_representatives(
                    data_subset, settings, target_alts[-1])
                print(record_time('end') + '\n', file=flog)
                
                predictions.to_pickle(f'{output_path}/predictions/{target}.pkl')
                print(f'\tPredictions saved')
