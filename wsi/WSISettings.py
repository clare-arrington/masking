from collections import namedtuple
ModelInfo = namedtuple('ModelInfo', ['name', 'language', 'vocab_size'])

models = {
    'bert' : ModelInfo(
        'bert-base-uncased', 
        'english', 30522
        ),
    'beto' : ModelInfo(
        'dccuchile/bert-base-spanish-wwm-uncased',
        'spanish', 31002
        )
}

## Swap this for a different model
model = models['bert']

WSISettings = namedtuple('WSISettings', [
    'cuda_device', 'init_num_senses', 'subset_num',
    'disable_tfidf', 'disable_lemmatization', 
    'bert_model', 'language',
    'max_batch_size', 'prediction_cutoff' ])

DEFAULT_PARAMS = WSISettings(
    ## Cutoff for the dendrogram based on last n merges
    init_num_senses=15,
    ## Number of term instances that will be used for clustering
    subset_num=10000,
    cuda_device=1,
    ## BERT settings
    disable_lemmatization=True,
    disable_tfidf=False,
    max_batch_size=32,
    language=model.language,
    prediction_cutoff=model.vocab_size,
    bert_model=model.name
)
