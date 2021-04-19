import argparse
import sys, os, time, pickle as pkl
import time
import torch

import src.model_utils
from src import utils, generation_utils as gen_utils
import src.metrics

if __name__ == '__main__':
    parser = utils.make_basic_parser()
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)

    print('*********Prompt size =', args.prompt_size)

    if not args.use_large_feats:
        raise ValueError('Need to use large feats')

    # check if have to run
    save_directory = f'./outputs/{utils.get_dataset_name_from_datapath(args.data_dir)}_{utils.get_model_basename(args.model_name)}'
    name = f'{args.datasplit}_p{args.top_p}_k{args.top_k}_t{args.temp}_seed{args.seed}'
    folder_name = f'{save_directory}/generations/basic'
    if os.path.isfile(f'{folder_name}/feats_{name}.pt'):
        print(f'File: {folder_name}/feats_{name}.pt already exists. Exiting')
        sys.exit(-1)
    else:
        print(f'File: {folder_name}/feats_{name}.pt does not exist. Proceeding with generation')


    device = utils.get_device_from_arg(args.device)
    print(f'Using device: {device}')

    model, tokenizer = utils.get_model_and_tokenizer(model_name=args.model_name, device=device)

    if args.max_len is None:
        args.max_len = tokenizer.model_max_length

    ds_tokens = utils.load_and_tokenize_data(tokenizer, args.data_dir, args.max_len, args.max_num_generations,
                                             min_len=args.prompt_size, split=args.datasplit)

    if os.path.isfile(f'{folder_name}/sample_{name}.p'):
        print(f'Undecoded samples: {folder_name}/sample_{name}.p already exist. Skipping generation.')
        with open(f'{folder_name}/sample_{name}.p', 'rb') as f:
            samples, is_completed, unique_ngram_frac, ppl = pkl.load(f)[:4]
        samples_2 = [torch.LongTensor(x).view(1, -1).to(device) for x in samples]
    else:
        batch_size = gen_utils.get_default_batch_size(args.model_name, device)
        n_lst = [1, 2, 3, 4, 5, 6]

        sample_fn = gen_utils.create_sample_fn(model, args.max_len,
            top_p=args.top_p, top_k=args.top_k, temperature=args.temp
        )
        t1 = time.time()
        samples, is_completed = gen_utils.get_samples_from_sample_fn(
            sample_fn, ds_tokens, tokenizer.eos_token_id,
            prompt_size=args.prompt_size, batch_size=batch_size
        )
        t2 = time.time()
        print('sampling time:', round(t2-t1, 2))
        unique_ngram_frac = src.metrics.get_unique_ngram_fraction(samples, n_lst)
        print('n-gram frac:', unique_ngram_frac)
        t1 = time.time()
        samples_2 = [torch.LongTensor(x).view(1, -1).to(device) for x in samples]
        ppl = src.metrics.get_perplexity_from_samples(model, samples_2)
        t2 = time.time()
        print('ppl time:', round(t2-t1, 2), ppl)

        output_file_name = f'{folder_name}/sample_{name}.p'  # un-decoded samples
        with open(output_file_name, 'wb') as f:
            pkl.dump([samples, is_completed, unique_ngram_frac, ppl, args], f)

    # decode samples
    print('Deocding...')
    if os.path.isfile(f'{folder_name}/sentences_{name}.p'):
        print(f'Decode samples: {folder_name}/sentences_{name}.p already exist. Skipping.')
    else:
        decoded_samples = utils.decode_samples_from_lst(tokenizer, samples)
        with open(f'{folder_name}/sentences_{name}.p', 'wb') as f:
            pkl.dump([decoded_samples, is_completed], f)

    # featurize samples
    print('Featurizing...')
    feats_prefix = ''
    if args.use_large_feats:
        del model
        model, _ = utils.get_model_and_tokenizer(model_name=args.featurize_model_name, device=device)
        for l in {128, 256, 512, args.max_len}:
            feats_prefix = f'L{l}'
            feats_out_fn = f'{folder_name}/feats{feats_prefix}_{name}.pt'
            if os.path.isfile(feats_out_fn):
                print(f'Feats {feats_out_fn} exisits. Skipping')
                continue
            else:
                print(f'Featurizing l = {l}...')
                samples_3 = [x[:, :l] for x in samples_2]
                feats = src.model_utils.featurize_sequential(model, samples_3)
                torch.save(feats, feats_out_fn)
    else:  # use features from model
        feats = src.model_utils.featurize_sequential(model, samples_2)
        torch.save(feats, f'{folder_name}/feats_{name}.pt')

