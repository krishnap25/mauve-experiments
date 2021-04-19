import random
import os
import pickle as pkl
import time
import torch
from nltk.translate.bleu_score import SmoothingFunction
from functools import partial
from multiprocessing.pool import Pool

import src.utils as utils
from src.generation_utils import self_bleu_one_sentence, get_bleu_weight_for_ngram
from src.utils import tqdm
import src.metrics


def main():
    parser = utils.make_metrics_parser()
    args = parser.parse_args()
    main_metrics(args)
    main_bleu(args)

# Pass args: datasplit, data_dir, parllel_bleu, n_proc_bleu
#  time python -u compute_ref_metrics.py --datasplit test --device 1 --parallel_bleu --n_proc_bleu 12 > outs/ref/test 2>&1

def main_metrics(args):
    device = utils.get_device_from_arg(args.device)
    print(f'Using device: {device}')

    save_directory = f'./outputs/{utils.get_dataset_name_from_datapath(args.data_dir)}_{utils.get_model_basename(args.model_name)}'
    model, tokenizer = utils.get_model_and_tokenizer(model_name=args.model_name, device=device)
    folder = 'ref'
    if args.ds_name is None:
        filename = args.datasplit
    else:
        filename = f'{args.ds_name}_{args.datasplit}'

    ds_tokens = utils.load_and_tokenize_data(
        tokenizer, args.data_dir, args.max_len, args.max_num_data,
        ds_name=args.ds_name, split=args.datasplit
    )
    savefilename = f'{save_directory}/metrics/{folder}/all_{filename}.p'
    if os.path.isfile(savefilename) and not args.force:
        print('All metrics already computed. Exiting')
        return

    all_sentences = [x[0].numpy().tolist() for x in ds_tokens]
    is_completed = [True for _ in all_sentences]

    metrics_all = {}

    # Distinct-n
    n_lst = [1, 2, 3, 4, 5, 6]
    unique_ngram_frac = src.metrics.get_unique_ngram_fraction(all_sentences, n_lst)
    metrics_all['distinct-n'] = unique_ngram_frac

    # PPL
    samples_2 = [torch.LongTensor(x).view(1, -1).to(device) for x in all_sentences]
    ppl = src.metrics.get_perplexity_from_samples(model, samples_2)
    metrics_all['perplexity'] = ppl

    # Zipf
    metrics_all['zipf'] = src.metrics.zipf_coeff(all_sentences)

    # Repetition
    metrics_all['repetition'] = src.metrics.get_repetition_fraction(all_sentences)

    # Non-termination
    metrics_all['non-termination-ratio'] = src.metrics.get_nontermination_ratio(all_sentences, is_completed)

    # save
    with open(savefilename, 'wb') as f:
        pkl.dump(metrics_all, f)
    print(f'Done. Saved "{savefilename}". Bye!')


def main_bleu(args):
    rng = random.Random(args.seed)

    save_directory = f'./outputs/{utils.get_dataset_name_from_datapath(args.data_dir)}_{utils.get_model_basename(args.model_name)}'
    _, tokenizer = utils.get_model_and_tokenizer(model_name=args.model_name, device=utils.CPU_DEVICE)
    folder = 'ref'
    if args.ds_name is None:
        filename = args.datasplit
    else:
        filename = f'{args.ds_name}_{args.datasplit}'

    ds_tokens = utils.load_and_tokenize_data(
        tokenizer, args.data_dir, args.max_len, args.max_num_data,
        ds_name=args.ds_name, split=args.datasplit
    )
    all_sentences = [x[0].numpy().tolist() for x in ds_tokens]

    savefilename = f'{save_directory}/metrics/{folder}/bleu_{filename}.p'
    if os.path.isfile(savefilename) and not args.force:
        print('Bleu metrics already computed. Exiting')
        return

    smoothing_function = SmoothingFunction().method1

    start_time = time.time()
    if args.parallel_bleu:
        bleu_scores = compute_bleus_parallel(all_sentences, smoothing_function, rng, args)
    else:
        bleu_scores = compute_bleus_sequential(all_sentences, smoothing_function, rng, args)
    print('Total time for self bleu:', round(time.time() - start_time), 's')


    # save
    with open(savefilename, 'wb') as f:
        pkl.dump(bleu_scores, f)
    print(f'Done. Saved "{savefilename}". Bye!')


def compute_bleus_sequential(all_sentences, smoothing_function, rng, args):
    bleu_scores = []
    for n in range(1, 6):
        start_time = time.time()
        weights = get_bleu_weight_for_ngram(n)
        bleu_n_lst = [
            self_bleu_one_sentence(weights, all_sentences, smoothing_function, i)
            for i in rng.sample(range(len(all_sentences)), min(len(all_sentences), args.n_sample_bleu))
        ]
        bleu_scores.append(sum(bleu_n_lst) / len(bleu_n_lst))
        print(f'Total time for self bleu-{n}:', round(time.time() - start_time), 's')
    return bleu_scores


def compute_bleus_parallel(all_sentences, smoothing_function, rng, args):
    pool = Pool(processes=min(args.n_proc_bleu, os.cpu_count()))
    bleu_scores = []
    for n in range(1, 6):
        start_time = time.time()
        weights = get_bleu_weight_for_ngram(n)
        bleu_n_lst = list(tqdm(
            pool.imap_unordered(
                partial(self_bleu_one_sentence, weights, all_sentences, smoothing_function),
                rng.sample(range(len(all_sentences)), min(len(all_sentences), args.n_sample_bleu))),
            total=args.n_sample_bleu))
        bleu_scores.append(sum(bleu_n_lst) / len(bleu_n_lst))
        print(f'Total time for self bleu-{n}:', round(time.time() - start_time), 's')
    return bleu_scores

if __name__ == '__main__':
    main()
