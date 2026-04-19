import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertForMaskedLM, BertTokenizer
from metrics import compute_surprisal, compute_attention_entropy, compute_bert_surprisal
import numpy as np

class GPT2Evaluator:
    def __init__(self, model_name="gpt2"):
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def evaluate_sentence(self, sentence: str):
        # We need word-to-token alignment.
        words = sentence.split()
        
        inputs = self.tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            
        logits = outputs.logits
        attentions = outputs.attentions
        
        # Token-level metrics
        token_surprisals = compute_surprisal(logits, input_ids)
        token_entropies = compute_attention_entropy(attentions)
        
        # Aggregate to word level
        word_metrics = self._aggregate_to_words(words, input_ids[0], token_surprisals, token_entropies)
        return word_metrics

    def _aggregate_to_words(self, words, input_ids, token_surprisals, token_entropies):
        # Simple alignment: greedy subword matching
        word_metrics = []
        token_idx = 0
        decoded_tokens = [self.tokenizer.decode([tok]).strip() for tok in input_ids]
        
        for w_idx, word in enumerate(words):
            subword_surprisal = 0.0
            subword_entropies = []
            
            # Very simplistic greedy alignment
            current_str = ""
            start_tok_idx = token_idx
            while token_idx < len(decoded_tokens) and len(current_str) < len(word):
                current_str += decoded_tokens[token_idx].replace("Ġ", "")
                subword_surprisal += token_surprisals[token_idx]
                subword_entropies.append(token_entropies[token_idx])
                token_idx += 1
                
            # Average entropies across subwords
            avg_entropies = {}
            if len(subword_entropies) > 0:
                keys = subword_entropies[0].keys()
                for k in keys:
                    avg_entropies[k] = np.mean([head_ent[k] for head_ent in subword_entropies])
            
            word_metrics.append({
                "word": word,
                "surprisal": subword_surprisal,
                "attention_entropy": avg_entropies
            })
            
        return word_metrics

class BertEvaluator:
    def __init__(self, model_name="bert-base-uncased"):
        print(f"Loading {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name, output_attentions=True)
        self.model.eval()

    def evaluate_sentence(self, sentence: str):
        words = sentence.split()
        
        inputs = self.tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        
        # Get attention entropy by doing a single pass
        with torch.no_grad():
            outputs = self.model(input_ids)
        attentions = outputs.attentions
        token_entropies = compute_attention_entropy(attentions)
        
        # Get Surprisal by masking token-by-token
        preds_list = []
        for i in range(seq_len):
            masked_input_ids = input_ids.clone()
            # Ignore masking [CLS] and [SEP]
            if i > 0 and i < seq_len - 1:
                masked_input_ids[0, i] = self.tokenizer.mask_token_id
            
            with torch.no_grad():
                mask_outputs = self.model(masked_input_ids)
            preds_list.append(mask_outputs.logits)
            
        token_surprisals = compute_bert_surprisal(preds_list, input_ids)
        
        # Aggregate to word level
        word_metrics = self._aggregate_to_words(words, input_ids[0], token_surprisals, token_entropies)
        return word_metrics

    def _aggregate_to_words(self, words, input_ids, token_surprisals, token_entropies):
        word_metrics = []
        # BERT tokens include [CLS] and [SEP]. We skip [CLS]
        token_idx = 1
        decoded_tokens = [self.tokenizer.decode([tok]).replace(" ", "") for tok in input_ids]
        
        for w_idx, word in enumerate(words):
            subword_surprisal = 0.0
            subword_entropies = []
            
            current_str = ""
            while token_idx < len(decoded_tokens) - 1 and len(current_str) < len(word):
                tok_str = decoded_tokens[token_idx].replace("##", "")
                current_str += tok_str
                subword_surprisal += token_surprisals[token_idx]
                subword_entropies.append(token_entropies[token_idx])
                token_idx += 1
                
            avg_entropies = {}
            if len(subword_entropies) > 0:
                keys = subword_entropies[0].keys()
                for k in keys:
                    avg_entropies[k] = np.mean([head_ent[k] for head_ent in subword_entropies])
            
            word_metrics.append({
                "word": word,
                "surprisal": subword_surprisal,
                "attention_entropy": avg_entropies
            })
            
        return word_metrics
