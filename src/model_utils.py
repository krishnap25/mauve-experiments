import argparse
import numpy as np
import math
import sys, os, time, pickle as pkl
import json
import random
from tqdm.auto import tqdm as tqdm_original
from typing import Optional
import torch
from torch.nn.functional import softmax, log_softmax, relu

from src.utils import tqdm


@torch.no_grad()
def my_top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
    NOTE: hidden state must be from prior to the token output at the logitso
        pass in first_token_p if reliable hidden state is not available
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = (cumulative_probs > top_p)
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def reshape_logit_scores(
        scores, temperature=1.0, top_k=0, top_p=1.0,
):
    # scores: (batch_size, length, vocab_size)
    assert temperature > 1e-10
    if temperature != 1.0:
        scores = scores / temperature
    shape = scores.shape
    # top_k_top_p_filtering requires 2D input
    scores = my_top_k_top_p_filtering(
        scores.view(-1, shape[-1]),
        top_k=top_k, top_p=top_p
    ).view(shape).contiguous()
    return scores


@torch.no_grad()
def get_tokenwise_log_probs_seq(
        model, sen, temperature=1.0, top_k=0, top_p=1.0,
):
    # TODO: only works for batch size 1
    device = next(model.parameters()).device
    sen = sen.to(device)
    outs = model(input_ids=sen, past_key_values=None,
         output_hidden_states=True, return_dict=True)
    logits = reshape_logit_scores(
        outs.logits, temperature, top_k, top_p,
    )
    log_probs = log_softmax(logits, dim=2)
    return log_probs  # (b, seq_len, vocab_size)


@torch.no_grad()
def get_log_probs_and_hidden_states(model, sen, hidden_layer=-1):
    device = next(model.parameters()).device
    sen = sen.to(device)
    outs = model(input_ids=sen, past_key_values=None,
         output_hidden_states=True, return_dict=True)
    log_probs = log_softmax(outs.logits, dim=2)  # (b, seq_len, vocab_size)
    hs = outs.hidden_states[hidden_layer]  # (b, seq_len, hidden_dim)
    return log_probs, hs


@torch.no_grad()
def get_logprob_of_seq_from_logits(logits, seq):
    # logits: (batch_size, seq_len, vocab_size)
    # works only if all elements in the batch have the same shape
    batch_size, seq_len = logits.shape[:2]
    log_probs = log_softmax(logits, dim=2)  # (b, seq_len, vocab_size)
    # seq_next = (seq[1], seq[2], ..., seq[-1])
    permutation = torch.arange(1, seq_len)
    seq_next = seq[:, permutation] # (batch_size, seq_len-1)
    # pick up log-probs corresponding to observed sequence
    i = torch.ger(torch.arange(batch_size), torch.ones(seq_len-1, dtype=torch.long))
    j = torch.ger(torch.ones(batch_size, dtype=torch.long), torch.arange(seq_len-1))
    return log_probs[i, j, seq_next] # (seq_len-1,)


@torch.no_grad()
def get_reshaped_log_probs_of_ds(model, ds_tokens, top_p=1.0, top_k=0, temperature=1.0):
    log_probs = []
    device = next(model.parameters()).device
    for sen in tqdm(ds_tokens):
        sen = sen.to(device)
        outs = model(input_ids=sen, past_key_values=None,
                     output_hidden_states=False, return_dict=True)
        logits = reshape_logit_scores(
            outs.logits, temperature, top_k, top_p,
        )
        # log_p: (seq_len-1,)
        log_p = get_logprob_of_seq_from_logits(logits, sen)
        log_probs.append(log_p.detach().cpu())
    return log_probs

@torch.no_grad()
def get_log_probs_of_ds(model, ds_tokens):
    log_probs = []
    device = next(model.parameters()).device
    num_tokens = 0
    for sen in ds_tokens:
        num_tokens += sen.view(-1).shape[0]
        sen = sen.to(device)
        outs = model(input_ids=sen, past_key_values=None,
             output_hidden_states=False, return_dict=True)
        # log_p: (seq_len,)
        log_p = get_logprob_of_seq_from_logits(outs.logits, sen)
        log_probs.append(log_p.sum(axis=1))
    return torch.cat(log_probs), num_tokens


@torch.no_grad()
def featurize_sequential(model, ds_tokens):
    device = next(model.parameters()).device
    t1 = time.time()
    feats = []
    for sen in tqdm(ds_tokens):
        sen = sen.to(device)
        outs = model(input_ids=sen, past_key_values=None,
             output_hidden_states=True, return_dict=True)
        h = outs.hidden_states[-1]  # (batch_size, seq_len, dim)
        feats.append(h[:, -1, :].cpu())
    t2 = time.time()
    print(f'Featurize time: {round(t2-t1, 2)}')
    return torch.cat(feats)


