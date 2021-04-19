import random
import os
import pickle as pkl
import time
from nltk.translate.bleu_score import SmoothingFunction
from functools import partial
from multiprocessing.pool import Pool

import src.utils as utils
from src.generation_utils import self_bleu_one_sentence, get_bleu_weight_for_ngram
from src.utils import tqdm

# Inspired by https://github.com/ari-holtzman/degen/blob/master/metrics/self_bleu.py

# Run time (serial): ~7 hours

def main():
    parser = utils.make_metrics_parser()
    args = parser.parse_args()
    rng = random.Random(args.seed)

    save_directory = f'./outputs/{utils.get_dataset_name_from_datapath(args.data_dir)}_{utils.get_model_basename(args.model_name)}'
    folder, filename = utils.get_save_filename_from_args(args)
    if not os.path.isfile(f'{save_directory}/generations/{folder}/sample_{filename}.p'):
        raise FileNotFoundError(f'Generations {save_directory}/generations/{folder}/sample_{filename}.p do not exist')

    savefilename = f'{save_directory}/metrics/{folder}/bleu_{filename}.p'
    if os.path.isfile(savefilename) and not args.force:
        print('Bleu metrics already computed. Exiting')
        return

    with open(f'{save_directory}/generations/{folder}/sample_{filename}.p', 'rb') as f:
        all_sentences = pkl.load(f)[0]
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
            for i in rng.sample(range(len(all_sentences)), args.n_sample_bleu)
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
                rng.sample(range(len(all_sentences)), args.n_sample_bleu)),
            total=args.n_sample_bleu))
        bleu_scores.append(sum(bleu_n_lst) / len(bleu_n_lst))
        print(f'Total time for self bleu-{n}:', round(time.time() - start_time), 's')
    return bleu_scores


if __name__ == '__main__':
    main()
