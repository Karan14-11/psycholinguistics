import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from metrics import compute_surprisal, compute_attention_entropy
import numpy as np


class GPT2Evaluator:
    """
    GPT-2 evaluator that computes token-level surprisal and per-layer/head
    attention entropy, then aggregates to word level.
    """

    def __init__(self, model_name="gpt2"):
        print(f"Loading {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.num_layers = self.model.config.n_layer   # 12 for gpt2
        self.num_heads = self.model.config.n_head     # 12 for gpt2
        print(f"  {model_name}: {self.num_layers} layers × {self.num_heads} heads")

    def evaluate_sentence(self, sentence: str):
        """
        Run a forward pass through GPT-2 and return word-level metrics.
        Each word metric dict contains:
            - word: str
            - surprisal: float (summed over sub-tokens)
            - attention_entropy: dict of {L{l}_H{h}: float}
        """
        words = sentence.split()

        inputs = self.tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Truncate if too long for GPT-2 (max 1024 tokens)
        if input_ids.shape[1] > 1024:
            input_ids = input_ids[:, :1024]

        with torch.no_grad():
            outputs = self.model(input_ids)

        logits = outputs.logits
        attentions = outputs.attentions  # tuple of (1, n_heads, seq_len, seq_len)

        # Token-level metrics
        token_surprisals = compute_surprisal(logits, input_ids)
        token_entropies = compute_attention_entropy(attentions)

        # Aggregate to word level
        word_metrics = self._aggregate_to_words(words, input_ids[0], token_surprisals, token_entropies)
        return word_metrics

    def _aggregate_to_words(self, words, input_ids, token_surprisals, token_entropies):
        """
        Maps sub-word token metrics back to word-level via greedy alignment.
        """
        word_metrics = []
        token_idx = 0
        decoded_tokens = [self.tokenizer.decode([tok]).strip() for tok in input_ids]

        for w_idx, word in enumerate(words):
            subword_surprisal = 0.0
            subword_entropies = []

            # Greedy subword alignment
            current_str = ""
            while token_idx < len(decoded_tokens) and len(current_str) < len(word):
                tok_text = decoded_tokens[token_idx].replace("Ġ", "").replace(" ", "")
                current_str += tok_text
                subword_surprisal += token_surprisals[token_idx]
                subword_entropies.append(token_entropies[token_idx])
                token_idx += 1

            # Average entropies across subwords for each head
            avg_entropies = {}
            if len(subword_entropies) > 0:
                keys = subword_entropies[0].keys()
                for k in keys:
                    avg_entropies[k] = float(np.mean([se[k] for se in subword_entropies]))

            word_metrics.append({
                "word": word,
                "surprisal": subword_surprisal,
                "attention_entropy": avg_entropies
            })

        return word_metrics
