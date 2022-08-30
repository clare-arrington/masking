from transformers import BertForMaskedLM, BertTokenizer
# from transformers import pipeline
from nltk.corpus import stopwords
from tqdm import tqdm
import multiprocessing
import numpy as np
import pandas as pd
import torch
import spacy
import re

def get_batches(from_iter, group_size):
    ret = []
    for _, x in from_iter:
        ret.append(x)
        if len(ret) == group_size:
            yield ret
            ret = []
    if ret:
        yield ret

def apply_softmax(values):
    e_x = np.exp(values - np.max(values))
    return e_x / e_x.sum()

def trim_predictions_count(
    likelihoods, language, n=50):
    stops = stopwords.words(language)
    stops.remove('no')
    shared_words = set()
    num_words = []
    for inst_id, probs in likelihoods.iterrows():
        nums = []
        for predicted_word, prob in probs.nlargest(500).iteritems():
            filtered_word = re.sub(r'[^a-z]', '', predicted_word)
            if len(filtered_word) <= 2 or filtered_word in stops:
                continue
             
            shared_words.add(predicted_word)
            nums.append(predicted_word)
            if len(nums) >= n:
                break
        num_words.append(len(nums))

    print(len(shared_words), sum(num_words)//len(num_words))
    print(num_words[:5])
    return likelihoods[shared_words]

def trim_predictions(
    likelihoods, targets, language, cutoff=1, threshold=.0005):
    stops = stopwords.words(language)
    stops.remove('no')
    stops.extend(targets)

    shared_words = set()
    num_words = []
    for inst_id, probs in likelihoods.iterrows(): 
        probs = probs.sort_values(ascending=False)

        cumulative_density = 0
        num = 0
        for predicted_word, prob in probs.iteritems():
            filtered_word = re.sub(r'[^a-z]', '', predicted_word)
            if len(filtered_word) <= 2 or filtered_word in stops:
                continue
             
            shared_words.add(predicted_word)
            cumulative_density += prob
            num += 1
            if cumulative_density >= cutoff or prob < threshold:
                break
        num_words.append(num)

    # print(len(shared_words), sum(num_words)//len(num_words))
    # print(num_words[:5])
    ## TODO: some columns are the same, should use numerical ids instead of words to select
    return shared_words

class LMBert():
    def __init__(self, settings):
        if settings.cuda_device >= 0:
            device = torch.device(f'cuda:{settings.cuda_device}')  
        else:
            torch.device('cpu')

        with torch.no_grad():
            model = BertForMaskedLM.from_pretrained(settings.bert_model, output_hidden_states=True)
            model.cls.predictions = model.cls.predictions.transform
            model.to(device=device)
            model.eval()
            self.bert = model
            self.device = device
            self.tokenizer = BertTokenizer.from_pretrained(
                settings.bert_model)

            self.max_sent_len = model.config.max_position_embeddings
            self.max_batch_size = settings.max_batch_size
            self.lemmatized_vocab = []
            self.original_vocab = []

            sp_models = {
                "english": "en_core_web_sm",
                "spanish": "es_core_news_sm"}

            nlp = spacy.load(sp_models[settings.language], 
                             disable=['ner', 'parser'])
            self._lemmas_cache = {}
            self._spacy = nlp
            for spacyed in tqdm(
                    nlp.pipe(self.tokenizer.vocab.keys(), 
                    batch_size=1000, n_process=multiprocessing.cpu_count()),
                    total=len((self.tokenizer.vocab)), 
                    desc='lemmatizing vocab'):
                lemma = spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_
                self._lemmas_cache[spacyed[0].lower_] = lemma
                self.lemmatized_vocab.append(lemma)
                self.original_vocab.append(spacyed[0].lower_)


    def format_sentence_to_pattern(self, pre, target, post, pattern):
        replacements = dict(pre=pre, target=target, post=post)
        for predicted_token in ['{mask_predict}', '{target_predict}']:
            if predicted_token in pattern: 
                before_pred, after_pred = pattern.split(predicted_token)
                before_pred = ['[CLS]'] + self.tokenizer.tokenize(before_pred.format(**replacements))
                after_pred = self.tokenizer.tokenize(after_pred.format(**replacements)) + ['[SEP]']
                target_prediction_idx = len(before_pred)
                target_tokens = ['[MASK]'] if predicted_token == '{mask_predict}' else self.tokenizer.tokenize(target)
                return before_pred + target_tokens + after_pred, target_prediction_idx

    def _get_lemma(self, word):
        if word in self._lemmas_cache:
            return self._lemmas_cache[word]
        else:
            spacyed = self._spacy(word)
            lemma = spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_
            self._lemmas_cache[word] = lemma
            return lemma

    def predict_sent_substitute_representatives(self, data_subset, settings, target):
        patterns = [('{pre} {target_predict} {post}', 1)]
        n_patterns = len(patterns)
        pattern_str, pattern_weights = list(zip(*patterns))
        pattern_weights = torch.from_numpy(np.array(pattern_weights, dtype=np.float32).reshape(-1, 1)).to(device=self.device)
        num_predictions = settings.prediction_cutoff

        with torch.no_grad():
            sorted_by_len = data_subset.sort_values(by="length")[['word_idx','formatted_sent']]
            inst_ids = []
            predictions = []

            for batch in get_batches(sorted_by_len.iterrows(),
                                     self.max_batch_size // n_patterns):

                # Converts the sentences to BERT format
                # Num patterns x num sentences
                batch_sents = []
                # Skip target here to use the passed in target instead
                for inst_id, (pre, _, post) in batch:
                    for pattern in pattern_str:
                        formatted_sent = self.format_sentence_to_pattern(pre, target, post, pattern)
                        batch_sents.append(formatted_sent)

                # Converts terms to BERT tokens
                tokenized_sents_vocab_idx = [self.tokenizer.convert_tokens_to_ids(sent[0]) for sent in batch_sents]

                # Right pads sentences to make all the same length
                max_len = max(len(x) for x in tokenized_sents_vocab_idx)
                batch_input = np.zeros((len(tokenized_sents_vocab_idx), max_len), dtype=np.int64)

                for idx, vals in enumerate(tokenized_sents_vocab_idx):
                    batch_input[idx, 0:len(vals)] = vals

                # Makes vectors into tensors
                torch_input_ids = torch.tensor(batch_input, dtype=torch.long).to(device=self.device)

                # TODO: input attention mask can be applied here
                torch_mask = torch_input_ids != 0

                # Logits: pred. scores (for each vocabulary token before SoftMax)
                pred_results = self.bert(torch_input_ids, attention_mask=torch_mask)
                logits_all_tokens = pred_results.logits
                #attention = pred_results.attentions

                # Select the logits for the masked term
                # Logit shape: 1 per sentence x 1 per word x 768 (hidden state size)
                logits_target_tokens = torch.zeros((len(batch_sents), logits_all_tokens.shape[2])).to(self.device)
                for i in range(0, len(batch_sents)):
                    logits_target_tokens[i, :] = logits_all_tokens[i, batch_sents[i][1], :]

                # Combine the multiple pattern versions of a sentence into one 
                logits_target_tokens_joint_patt = torch.zeros(
                    (len(batch_sents) // n_patterns, logits_target_tokens.shape[1])).to(
                    self.device)
                    
                for i in range(0, len(batch_sents), n_patterns):
                    logits_target_tokens_joint_patt[i // n_patterns, :] = (
                            logits_target_tokens[i:i + n_patterns, :] * pattern_weights).sum(0)

                # Softmax is applied to the vocab to get the probs 
                pre_softmax = torch.matmul(
                logits_target_tokens_joint_patt,
                self.bert.bert.embeddings.word_embeddings.weight.transpose(0, 1))

                # Get top terms for each sentence
                topk_vals, topk_idxs = torch.topk(pre_softmax, num_predictions, -1)

                # Apply softmax to logits
                probs_batch = torch.softmax(topk_vals, -1).detach().cpu().numpy()
                topk_idxs_batch = topk_idxs.detach().cpu().numpy()

                for (inst_id, _), probs, topk_idxs in zip(batch, probs_batch, topk_idxs_batch):
                    inst_ids.append(inst_id)
                    predictions.append({idx : prob for idx, prob in zip(topk_idxs, probs)})

        predictions = pd.DataFrame(data=predictions, index=inst_ids)
        predictions.head()

        ## Reorder and rename columns
        predictions = predictions[range(num_predictions)]

        # Lemmatized vocab is enabled by default
        # That means we use BERT's 30522 vocab
        # Or BETO's 31002
        if settings.disable_lemmatization:
            predictions.columns = self.original_vocab  
        else:
            predictions.columns = self.lemmatized_vocab

        return predictions

    def get_embedded_sents(self, data_subset, target):
        pattern_str = ('{pre} {target_predict} {post}',)

        with torch.no_grad():
            sorted_by_len = data_subset.sort_values(by="length")[['word_idx','formatted_sent']]
            vectors = {}

            for batch in get_batches(sorted_by_len.iterrows(),
                                     self.max_batch_size):
 
                # Converts the sentences to BERT format
                # Num patterns x num sentences
                batch_sents = []
                target_locs = {}
                for inst_id, (pre, _, post) in batch:
                    for pattern in pattern_str:
                        formatted_sent = self.format_sentence_to_pattern(pre, target, post, pattern)
                        batch_sents.append(formatted_sent[0])
                        target_locs[inst_id] = formatted_sent[1]

                # Converts terms to BERT tokens
                tokenized_sents_vocab_idx = [self.tokenizer.convert_tokens_to_ids(sent) for sent in batch_sents]

                # Right pads sentences to make all the same length
                max_len = max(len(x) for x in tokenized_sents_vocab_idx)
                batch_input = np.zeros((len(tokenized_sents_vocab_idx), max_len), dtype=np.int64)
                for idx, vals in enumerate(tokenized_sents_vocab_idx):
                    batch_input[idx, 0:len(vals)] = vals

                # Makes vectors into tensors
                torch_input_ids = torch.tensor(batch_input, dtype=torch.long).to(device=self.device)

                # TODO: input attention mask could be applied here
                torch_mask = torch_input_ids != 0

                # Logits: pred. scores (for each vocabulary token before SoftMax)
                pred_results = self.bert(torch_input_ids, attention_mask=torch_mask)
                hidden_states = [hs.detach().cpu().clone().numpy() for hs in pred_results.hidden_states]
                #.to(self.device).detach().cpu()
                
                # get usage vectors from hidden states
                hidden_states = np.stack(hidden_states)  # (13, B, |s|, 768)
                # print(f'Expected hidden states size: (13, {len(batch)}, |{max_len}|, 768)')
                # print('Got {}'.format(hidden_states.shape))
                usage_vectors = np.sum(hidden_states[1:, :, :, :], axis=0)
                
                ## Separate the hidden states by instance
                for inst_num, (inst_id, target_loc) in enumerate(target_locs.items()):
                    usage_vector = usage_vectors[inst_num, target_loc+1, :]
                    vectors[inst_id] = usage_vector
                    
        return vectors
