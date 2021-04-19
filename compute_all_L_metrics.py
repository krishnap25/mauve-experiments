import os
import pickle as pkl
import torch
import src.utils as utils
import src.metrics


def main():
    parser = utils.make_metrics_parser()
    args = parser.parse_args()
    main_metrics(args)

def main_metrics(args):
    print(f'device: {args.device}')
    device = utils.get_device_from_arg(args.device)
    print(f'Using device: {device}')

    save_directory = f'./outputs/{utils.get_dataset_name_from_datapath(args.data_dir)}_{utils.get_model_basename(args.model_name)}'
    filename = f'{args.datasplit}_p{args.top_p}_k{args.top_k}_t{args.temp}_seed{args.generate_seed}'
    folder_name = f'{save_directory}/generations/basic'


    input_file_name = f'{folder_name}/sample_{filename}.p'
    if not os.path.isfile(input_file_name):
        print(f'File {input_file_name} does not exist. Quitting!')
        return
    with open(input_file_name, 'rb') as f:
        all_sentences, is_completed = pkl.load(f)[:2]

    savefilename = f'{save_directory}/metrics/basic/all_L_{filename}.p'
    if os.path.isfile(savefilename) and not args.force:
        print('All metrics already computed. Exiting')
        return

    model, tokenizer = utils.get_model_and_tokenizer(model_name='gpt2-large', device=device)

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


if __name__ == '__main__':
    main()
