from collections import namedtuple

WSISettings = namedtuple('WSISettings', ['cuda_device', 'max_number_senses',
                                         'disable_tfidf', 'disable_lemmatization', 
                                         'min_sense_instances', 'bert_model',
                                         'max_batch_size', 'prediction_cutoff' 
                                         ])

DEFAULT_PARAMS = WSISettings(
    max_number_senses=15,
    min_sense_instances=50,
    # sense clusters that dominate less than this number of samples
    # would be remapped to their closest big sense

    cuda_device=0,
    disable_lemmatization=False,
    disable_tfidf=False,
    max_batch_size=10,
    prediction_cutoff=30522,
    bert_model='bert-base-uncased'
)
