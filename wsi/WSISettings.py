from collections import namedtuple

WSISettings = namedtuple('WSISettings', ['n_represents', 'n_samples_per_rep', 'cuda_device',
                                         'disable_tfidf', 'disable_lemmatization', 
                                         'min_sense_instances', 'bert_model',
                                         'max_batch_size', 'prediction_cutoff', 'max_number_senses'
                                         ])

DEFAULT_PARAMS = WSISettings(
    n_represents=15, ##TODO: where is this used?
    n_samples_per_rep=50,

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
