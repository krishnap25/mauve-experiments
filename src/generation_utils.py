from typing import Iterable, Optional

from nltk.translate.bleu_score import sentence_bleu

import src.model_utils
from src.metrics import tqdm

import torch
from torch.nn import functional as F
from transformers.file_utils import ModelOutput
from transformers.utils import logging

import src.utils as utils
from src.transformers_utils import postprocess_next_token_scores


logger = logging.get_logger(__name__)



@torch.no_grad()
def generate_text_from_recalibrated_model(
        model,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs) -> torch.LongTensor:
    r"""
    Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
    beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

    Arguments:
        same as Huggingface Transformers
    Return:

        :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
        The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
        shorter if all batches finished early due to the :obj:`eos_token_id`.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)  # added extra; not in the original HF Transformers

    # We cannot generate if the model does not have a LM head
    if model.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
        )

    max_length = max_length if max_length is not None else model.config.max_length
    min_length = min_length if min_length is not None else model.config.min_length
    do_sample = do_sample if do_sample is not None else model.config.do_sample
    early_stopping = early_stopping if early_stopping is not None else model.config.early_stopping
    use_cache = use_cache if use_cache is not None else model.config.use_cache
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    # temperature = temperature if temperature is not None else model.config.temperature
    # top_k = top_k if top_k is not None else model.config.top_k
    # top_p = top_p if top_p is not None else model.config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else model.config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    length_penalty = length_penalty if length_penalty is not None else model.config.length_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else model.config.no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else model.config.bad_words_ids
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
    )
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else model.config.decoder_start_token_id
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
    assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
    ), "`no_repeat_ngram_size` should be a positive integer."
    assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."
    assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
    ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    # not allow to duplicate outputs when greedy decoding
    if do_sample is False:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                    num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                    num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    # create attention mask if necessary
    # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after
    # attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
        )
        pad_token_id = eos_token_id

    # vocab size
    if hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
    elif (
            model.config.is_encoder_decoder
            and hasattr(model.config, "decoder")
            and hasattr(model.config.decoder, "vocab_size")
    ):
        vocab_size = model.config.decoder.vocab_size
    else:
        raise ValueError("either model.config.vocab_size or model.config.decoder.vocab_size needs to be defined")

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if model.config.is_encoder_decoder:
        if decoder_start_token_id is None:
            # see if BOS token can be used for decoder_start_token_id
            if bos_token_id is not None:
                decoder_start_token_id = bos_token_id
            elif (
                    hasattr(model.config, "decoder")
                    and hasattr(model.config.decoder, "bos_token_id")
                    and model.config.decoder.bos_token_id is not None
            ):
                decoder_start_token_id = model.config.decoder.bos_token_id
            else:
                raise ValueError(
                    "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                )

        assert hasattr(model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(model)
        assert callable(model.get_encoder), "{} should be a method".format(model.get_encoder)

        # get encoder and store encoder outputs
        encoder = model.get_encoder()
        encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)

    # Expand input ids if num_beams > 1 or num_return_sequences > 1
    if num_return_sequences > 1 or num_beams > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, effective_batch_mult * num_beams, input_ids_len
        )

        input_ids = input_ids.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    if model.config.is_encoder_decoder:
        # create empty decoder input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        cur_len = 1

        assert (
                batch_size == encoder_outputs.last_hidden_state.shape[0]
        ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
        )

        # expand encoder_outputs
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_batch_idxs
        )

        # save encoder_outputs in `model_kwargs`
        model_kwargs["encoder_outputs"] = encoder_outputs

    else:
        cur_len = input_ids.shape[-1]

    assert (
            cur_len < max_length
    ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

    if num_beams > 1:
        raise ValueError('Cannot handle num_beams > 1')
    else:
        output = _generate_no_beam_search(model, input_ids,
                                          cur_len=cur_len, max_length=max_length, min_length=min_length,
                                          do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
                                          repetition_penalty=repetition_penalty,
                                          no_repeat_ngram_size=no_repeat_ngram_size, bad_words_ids=bad_words_ids,
                                          pad_token_id=pad_token_id, eos_token_id=eos_token_id,
                                          batch_size=effective_batch_size, attention_mask=attention_mask,
                                          use_cache=use_cache, model_kwargs=model_kwargs)

    return output


def _generate_no_beam_search(model, input_ids,
                             cur_len, max_length, min_length, do_sample,
                             temperature, top_k, top_p,
                             repetition_penalty, no_repeat_ngram_size, bad_words_ids,
                             pad_token_id, eos_token_id, batch_size, attention_mask, use_cache, model_kwargs):
    """Generate sequences for each example without beam search (num_beams == 1).
    All returned sequence are generated independently.
    """
    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    past = None
    while cur_len < max_length:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask,
            use_cache=use_cache, **model_kwargs
        )

        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
        # logits: (batch size, 1, vocab size); hidden state: (batch size, 1, dim)
        next_token_logits = outputs.logits[:, -1, :]

        scores = postprocess_next_token_scores(
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
        )

        # if model has past, then set the past variable to speed up decoding
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = src.model_utils.my_top_k_top_p_filtering(
                scores,
                top_k=top_k, top_p=top_p,)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if model.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    return input_ids


def batch_fn(iterable, n=1):
    # n: batch size
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:min(i+n, l)]

def get_default_batch_size(model_name, device, beam_size=1):
    # heuristic to figure out max batch size for model based on GPU memory
    twelve_gigs = 11719409664

    if 'gpt2-large' in model_name:
        default_batch_size = 8
    elif 'gpt2-xl' in model_name:
        default_batch_size = 2
    elif 'gpt2-medium' in model_name:
        default_batch_size = 20
    elif 'gpt2' in model_name:
        default_batch_size = 20
    else:
        # default_batch_size = 1
        raise ValueError(f'Unknown model {model_name}')
    if device == torch.device('cpu'):
        bsz = default_batch_size
    else:
        mem = torch.cuda.get_device_properties(device).total_memory
        bsz = int(mem / twelve_gigs * max(1, default_batch_size / beam_size))
        bsz = max(1, bsz)
    return bsz

def create_sample_fn(model, max_len,
                     top_p=1.0, top_k=0, temperature=1.0,
                     return_predicted_p=False):
    # recalib_fn is applied after top-p/top-k/temp modifications
    fn = lambda prompt: generate_text_from_recalibrated_model(
        model, input_ids=prompt,
        max_length=max_len, do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,)
    return fn

def remove_eos_from_samples(samples, eos_token_id):
    # samples: list of lists
    remove_eos_fn = lambda l: [x for x in l if x != eos_token_id]
    check_completed_fn = lambda l: any([x == eos_token_id for x in l])
    new_samples = [remove_eos_fn(s) for s in samples]
    is_completed = [check_completed_fn(s) for s in samples]
    return new_samples, is_completed

@torch.no_grad()
def get_samples_from_sample_fn(sampl_fn, ds_tokens, eos_token_id, prompt_size=10, batch_size=20):
    outs = []
    b_valid_ds = list(batch_fn(ds_tokens, batch_size))
    for b in tqdm(b_valid_ds):
        prompt = torch.cat([sen[:, :prompt_size] for sen in b])
        sample = sampl_fn(prompt)
        outs.extend(sample.cpu().numpy().tolist())
    # remove eos tokens which are added for padding
    outs, is_completed = remove_eos_from_samples(outs, eos_token_id)
    return outs, is_completed


def self_bleu_one_sentence(weights, all_sentences, smoothing_function, i):
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)


def get_bleu_weight_for_ngram(n_gram):
    if n_gram == 1:
        weights = (1.0, 0, 0, 0)
    elif n_gram == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n_gram == 3:
        weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
    elif n_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    elif n_gram == 5:
        weights = (0.2, 0.2, 0.2, 0.2, 0.2)
    else:
        raise ValueError
    return weights