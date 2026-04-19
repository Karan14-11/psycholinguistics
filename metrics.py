import torch
import numpy as np
from typing import List, Dict

def compute_surprisal(logits: torch.Tensor, target_ids: torch.Tensor) -> List[float]:
    """
    Computes S(w_i) = -log P(w_i | w_<i) for causal language models.
    logits: (batch_size, sequence_length, vocab_size)
    target_ids: (batch_size, sequence_length)
    """
    # Shift logits and targets for causal LM (Next token prediction)
    # P(w_i | w_<i) corresponds to the i-1'th logit predicting i'th token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    # Flatten inner dimensions for loss calculation
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Return as list of floats
    surprisals = loss.tolist()
    # Prepend a 0 for the first token since it has no prior context
    surprisals.insert(0, 0.0) 
    return surprisals

def compute_attention_entropy(attention_matrices: tuple) -> List[Dict[str, float]]:
    """
    Computes Attention Entropy H(A) = - sum A_ij log A_ij.
    attention_matrices: tuple of len(num_layers), each tensor of shape (1, num_heads, seq_len, seq_len)
    Returns: a list over sequence length of the attention entropy details per layer and head.
    """
    seq_len = attention_matrices[0].shape[-1]
    num_layers = len(attention_matrices)
    num_heads = attention_matrices[0].shape[1]
    
    # We find the entropy of the attention distribution at each token w_i 
    
    entropies = [] # list of seq_len length, each is a dict of {"L{layer}_H{head}": entropy}
    
    for i in range(seq_len):
        token_entropies = {}
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                # Get the attention distribution for token i
                # For causal models, it only looks at j <= i
                attn_dist = attention_matrices[layer_idx][0, head_idx, i, :].detach()
                
                # Filter out zeroes (e.g. padding/causal masking) to avoid nan in log
                attn_dist = attn_dist[attn_dist > 0]
                
                # H = -sum(p * log(p))
                if len(attn_dist) > 0:
                    entropy = -(attn_dist * torch.log(attn_dist)).sum().item()
                else:
                    entropy = 0.0
                token_entropies[f"L{layer_idx}_H{head_idx}"] = entropy
        entropies.append(token_entropies)
        
    return entropies
    
def compute_bert_surprisal(preds_list: List[torch.Tensor], target_ids: torch.Tensor) -> List[float]:
    """
    Computes pseudo-surprisal for Masked Language Models.
    Since MLMs are bidirectional, we compute Surprisal for token i by masking token i.
    preds_list: list of logits for each token position when that position was masked.
    """
    surprisals = []
    seq_len = target_ids.shape[1]
    loss_fct = torch.nn.CrossEntropyLoss()
    
    for i in range(seq_len):
        # preds_list[i] has shape (1, seq_len, vocab_size)
        # target is target_ids[0, i]
        logit = preds_list[i][0, i].unsqueeze(0) # shape (1, vocab_size)
        target = target_ids[0, i].unsqueeze(0)   # shape (1,)
        
        loss = loss_fct(logit, target)
        surprisals.append(loss.item())
        
    return surprisals
