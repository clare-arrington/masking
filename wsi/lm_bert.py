import multiprocessing
from transformers import BertForMaskedLM, BertTokenizer
import torch
import spacy
import numpy as np
from tqdm import tqdm
import pandas as pd

def get_batches(from_iter, group_size):
    ret = []
    for _, x in from_iter:
        ret.append(x)
        if len(ret) == group_size:
            yield ret
            ret = []
    if ret:
        yield ret

class LMBert():
    def __init__(self, cuda_device, bert_model, max_batch_size=20):
        device = torch.device(f'cuda:{cuda_device}') if cuda_device >= 0 else torch.device('cpu')

        with torch.no_grad():
            model = BertForMaskedLM.from_pretrained(bert_model)
            model.cls.predictions = model.cls.predictions.transform
            model.to(device=device)
            model.eval()
            self.bert = model

            self.tokenizer = BertTokenizer.from_pretrained(bert_model)

            self.max_sent_len = model.config.max_position_embeddings
            self.max_batch_size = max_batch_size
            self.lemmatized_vocab = []
            self.original_vocab = []

            nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
            self._lemmas_cache = {}
            self._spacy = nlp
            for spacyed in tqdm(
                    nlp.pipe(self.tokenizer.vocab.keys(), batch_size=1000, n_process=multiprocessing.cpu_count()),
                    total=len((self.tokenizer.vocab)), desc='lemmatizing vocab'):
                lemma = spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_
                self._lemmas_cache[spacyed[0].lower_] = lemma
                self.lemmatized_vocab.append(lemma)
                self.original_vocab.append(spacyed[0].lower_)

            self.device = device

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

    def predict_sent_substitute_representatives(self, data_subset, settings,):
        # TODO: why .5
        patterns = [('{pre} {target_predict} {post}', 0.5)]
        n_patterns = len(patterns)
        pattern_str, pattern_w = list(zip(*patterns))
        pattern_w = torch.from_numpy(np.array(pattern_w, dtype=np.float32).reshape(-1, 1)).to(device=self.device)
        num_predictions = 30522

        with torch.no_grad():
            
            sorted_by_len = data_subset.sort_values(by="length")[['word_index', 'formatted_sentence']]
            inst_ids = []
            predictions = []

            for batch in get_batches(sorted_by_len.iterrows(),
                                     self.max_batch_size // n_patterns):

                # Converts the sentences to BERT format
                # Num patterns x num batch_sents
                batch_sents = []
                for inst_id, (pre, target, post) in batch:
                    for pattern in pattern_str:
                        batch_sents.append(self.format_sentence_to_pattern(pre, target, post, pattern))

                # Converts terms to BERT tokens
                tokenized_sents_vocab_idx = [self.tokenizer.convert_tokens_to_ids(x[0]) for x in batch_sents]

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
                # TODO: why .8 for 2 sentences and .5 for one sentence total instead of 1?
                logits_target_tokens_joint_patt = torch.zeros(
                    (len(batch_sents) // n_patterns, logits_target_tokens.shape[1])).to(
                    self.device)
                    
                for i in range(0, len(batch_sents), n_patterns):
                    logits_target_tokens_joint_patt[i // n_patterns, :] = (
                            logits_target_tokens[i:i + n_patterns, :] * pattern_w).sum(0)

                # Softmax is applied to the vocab to get the probs 
                # Needed to structrue appropriately
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

        ## Reorder and rename columns
        predictions = predictions[range(num_predictions)]
        predictions.columns = self.original_vocab 
        
        # Lemmatized vocab is enabled by default
        # That means we use BERT's 30522 vocab
        #target_vocab = self.original_vocab if settings.disable_lemmatization else self.lemmatized_vocab


        return predictions
