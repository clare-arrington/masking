from collections import namedtuple

WSISettings = namedtuple('WSISettings', ['cuda_device', 'max_number_senses',
                                         'disable_tfidf', 'disable_lemmatization', 
                                         'min_sense_instances', 'bert_model',
                                         'max_batch_size', 'prediction_cutoff' 
                                         ])

DEFAULT_PARAMS = WSISettings(
    ## Cutoff for the dendrogram based on last n merges
    max_number_senses=15,

    ## Sense clusters that dominate less than this number of samples
    # would be remapped to their closest big sense
    # 25 - 100 range is good, but can scale up depending on size of data
    min_sense_instances=100,

    ## General BERT settings
    cuda_device=0,
    disable_lemmatization=False,
    disable_tfidf=False,
    max_batch_size=10,
    prediction_cutoff=30522,
    bert_model='bert-base-uncased'
)
