from collections import namedtuple

WSISettings = namedtuple('WSISettings', ['n_represents', 'n_samples_per_rep', 'cuda_device',
                                         'disable_tfidf', 'disable_lemmatization', 
                                         'min_sense_instances', 'bert_model',
                                         'max_batch_size', 'prediction_cutoff', 'max_number_senses'
                                         ])

DEFAULT_PARAMS = WSISettings(
    n_represents=15,
    n_samples_per_rep=20,
    cuda_device=0,
    disable_lemmatization=False,
    disable_tfidf=False,

    ## Patterns
    # (pattern,weight): each of these patterns will produce a prediction state.
    # the weighted sum of them will be matmul'ed for a distribution over substitutes

    max_number_senses=15,
    min_sense_instances=15,
    # sense clusters that dominate less than this number of samples
    # would be remapped to their closest big sense

    max_batch_size=10,
    prediction_cutoff=30522,
    bert_model='bert-base-uncased'
)
