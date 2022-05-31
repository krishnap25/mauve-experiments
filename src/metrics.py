import numpy as np
import operator
from sklearn import linear_model
import torch

import src.model_utils
from src.utils import tqdm
from nltk.util import ngrams as ngrams_fn_nltk
from collections import Counter

import src.utils as utils

######################
# Sparsemax score
######################
def sp_score_1(p, sen):
    # logp: (b, seq_len, vocab_size)
    n = p.shape[1]  # seq_len
    p = p[0, :-1, :]  # (n - 1, vocab_size)
    labels = sen[0, 1:]  # (n - 1)
    count = labels.shape[0]
    sp = p[torch.arange(n-1), labels] + 0.5 * (1 - torch.norm(p, dim=1)**2)  # (n-1)
    return sp.sum().item(), count

######################
# Jensen-Shannon score
######################

def kl(p, q):
    idxs = (p != 0)
    return (p[idxs] * torch.log(p[idxs] / q[idxs])).sum()

def js_score_1_naive(p, sen):
    # not numerically stable; direct implementation from the formula
    n = p.shape[1]  # seq_len
    p = p[0, :-1, :]  # (n - 1, vocab_size)
    labels = sen[0, 1:]  # (n - 1)
    count = labels.shape[0]
    p_true = torch.zeros_like(p)
    p_true[torch.arange(n-1), labels] = 1.0  # one hot
    m = 0.5 * (p + p_true)
    js = 0.5 * kl(p, m) + 0.5 * kl(p_true, m)
    return js.item(), count

def js_score_1(p, sen):
    # numerically stable version of JS score
    n = p.shape[1]  # seq_len
    p = p[0, :-1, :]
    logp = torch.log(p)  # (n - 1, vocab_size)
    labels = sen[0, 1:]  # (n - 1)
    count = labels.shape[0] # (n-1)
    idxs = torch.arange(n-1)
    p_true = p[idxs, labels]  # (n-1)
    logp_true = logp[idxs, labels]
    js1 =  np.log(2) + torch.where(p_true > 0,
           p_true * (logp_true - torch.log1p(p_true)),
            torch.zeros_like(p_true)
          )
    js2 = np.log(2) - torch.log1p(p[idxs, labels])
    return (js1 + js2).sum().item() * 0.5, count

######################
# eps-perplexity score
######################
def eps_perplexity(p, sen, eps, vocab_size):
    n = p.shape[1]  # seq_len
    gold_probs = p[0, torch.arange(n-1), sen[0, 1:]]
    return torch.log(gold_probs + eps) - np.log(1 + eps * vocab_size), n-1

def eps_perplexity_lst(p, sen, eps_lst, vocab_size):
    n = p.shape[1]  # seq_len
    gold_probs = p[0, torch.arange(n-1), sen[0, 1:]]
    ppl = (torch.log(gold_probs[None, :] + eps_lst[:, None])
           - torch.log(1 + eps_lst[:, None] * vocab_size)).sum(dim=1)
    return ppl, n-1

#######################################
# Repetition Statistics of Greedy Token
#######################################
def rep_score_1(p, sen, hist_size):
    p = p[0, :-1, :]  # (n-1, vocab_size)
    labels = sen[0, 1:]  # (n-1)
    count = labels.shape[0]
    greedy = p.argmax(dim=1) # (n-1)
    reps = sum([1 for i in range(labels.shape[0])
                if greedy[i] in labels[max(0, i-hist_size):i]])
    return reps, count

def wrep_score_1(p, sen, hist_size):
    p = p[0, :-1, :]  # (n-1, vocab_size)
    labels = sen[0, 1:]  # (n-1)
    count = labels.shape[0]
    greedy = p.argmax(dim=1) # (n-1)\
    reps = sum([1 for i in range(labels.shape[0])
                if (greedy[i] in labels[max(0, i-hist_size):i]
                    and greedy[i] != labels[i])
               ])
    return reps, count


#######################################################################
# Compute Metrics Based on Performance of Recalibrated Model on Dev Set
#######################################################################
def compute_metrics_from_probs(
        model, dataset, metric_fn_lst, eppl_eps_lst=[],
        temperature=1.0, top_k=0, top_p=1.0,
        vocab_size=50257
):
    l = len(metric_fn_lst)
    num_metrics = len(metric_fn_lst) + len(eppl_eps_lst)
    device = next(model.parameters()).device
    eppl_eps_lst = torch.from_numpy(np.asarray(eppl_eps_lst)).to(device)
    m_numer = np.zeros(num_metrics)
    m_denom = np.zeros(num_metrics)
    metrics_final = np.zeros(num_metrics)
    device = next(model.parameters()).device
    for sen in utils.tqdm(dataset):
        sen = sen.to(device)
        logp = src.model_utils.get_tokenwise_log_probs_seq(
            model, sen, temperature=temperature, top_k=top_k, top_p=top_p,)
        p = torch.exp(logp)
        for i, fn in enumerate(metric_fn_lst):
            m, c = fn(p, sen)
            m_numer[i] += m
            m_denom[i] += c
        if eppl_eps_lst.shape[0] > 0:
            e_ppl, c = eps_perplexity_lst(p, sen, eppl_eps_lst, vocab_size)
            m_numer[l:] += e_ppl.cpu().numpy()
            m_denom[l:] += c
    metrics_final[:l] = m_numer[:l] / m_denom[:l]  # fraction metrics
    metrics_final[l:] = np.exp(-m_numer[l:] / m_denom[l:])  # perplexity metrics
    return metrics_final

def get_probs_metric_fn_lst(ls=[64]):
    reps = [lambda p, s: rep_score_1(p, s, l) for l in ls]
    wreps = [lambda p, s: wrep_score_1(p, s, l) for l in ls]
    return [sp_score_1, js_score_1, *reps, *wreps]

def get_metric_names(ls=[64]):
    reps = [f'rep-{l}' for l in ls]
    wreps = [f'wrep-{l}' for l in ls]
    return ['sp-score', 'js-score', *reps, *wreps]


#######################################
# Disctinct-n metrics
#######################################
def get_ngram_freqs(samples, n):
    ngram_freq = Counter()
    for sen in samples:
        ngrams = ngrams_fn_nltk(sen, n)
        ngram_freq.update(ngrams)
    uniq = len(ngram_freq)
    total = sum(ngram_freq.values())
    return uniq, total

def get_unique_ngram_fraction(samples, n_lst):
    # distinct-n
    out = []
    for n in n_lst:
        a, b = get_ngram_freqs(samples, n)
        freq = a * 1.0 / b if b > 0 else 0
        out.append(freq)
    return out

#######################################
# Perplexity of Generations
#######################################
def _get_perplexity_from_prob(logp, num_tokens):
    return torch.exp(-logp.sum() / num_tokens).item()


def get_perplexity_from_samples(model, ds_tokens):
    logp, num_tokens = src.model_utils.get_log_probs_of_ds(model, ds_tokens)
    return _get_perplexity_from_prob(logp, num_tokens)

#######################################
# Zipf Coefficient
#######################################
def zipf_coeff(samples, min_num=1, max_num=5000, stretch_factor=15):
    # samples: list of lists of tokens; max_num: how many top frequency words to consider
    counter = Counter()
    for s in samples:
        counter.update(s)
    top_freqs = np.array(sorted(counter.values(), key=operator.neg)[:max_num])
    # log scale overweights tail, so subsample the tail
    # this also helps the best-fit line look more reasonable when plotted in log-scale.
    xs, idxs_u = np.unique(np.round(
        stretch_factor * np.log(np.arange(min_num, min(len(counter), max_num)).astype(np.float64))) / stretch_factor,
                           return_index=True)
    ys = np.log(top_freqs[idxs_u])

    lr = linear_model.LinearRegression()
    lr.fit(xs.reshape(-1, 1), ys)
    slope = lr.coef_[0]

    return slope

#######################################
# Repetition
#######################################
def get_repetition_fraction(samples, max_n=500):
    # from https://github.com/ari-holtzman/degen/blob/master/metrics/repetition.py
    n_repeated_examples = 0
    for gen in samples:
        rev_gen = list(reversed(gen))
        last_n_repeats = [0] * max_n
        for n in range(1, max_n + 1):
            n_repeat = 1
            while len(rev_gen[n * n_repeat:n * (n_repeat + 1)]) == n and \
                    rev_gen[n * n_repeat:n * (n_repeat + 1)] == rev_gen[:n]:
                n_repeat += 1
            last_n_repeats[n-1] = n_repeat
        max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])
        if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n+1 >= 3 or last_n_repeats[max_repeated_n] > 50):
            # repetition detected
            n_repeated_examples += 1
    return n_repeated_examples / len(samples)

#######################################
# Non-termination Ratio
#######################################
def get_nontermination_ratio(samples, is_completed):
    return sum(is_completed) / len(is_completed)
