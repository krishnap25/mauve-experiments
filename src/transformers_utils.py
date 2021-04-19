from typing import Iterable, List
import torch

##############################
# Legacy transformers code
##############################
def postprocess_next_token_scores(
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            scores,
            batch_size,
            num_beams,
            input_ids,
            repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # Exclude EOS token (already processed)
        bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
        # Modify the scores in place by setting the banned tokens logits to `-inf`
        set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

    return scores

def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """
    Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
    """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def calc_banned_ngram_tokens(prev_input_ids: torch.Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens

def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens

def set_scores_to_inf_for_banned_tokens(scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
    """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
    # [ 0  1  1 ]
    # [ 0  0  0 ]
    # [ 1  0  0 ]

    banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    scores.masked_fill_(banned_mask, -float("inf"))
