import os
import torch

import src.model_utils
from src import utils
import src.metrics

if __name__ == '__main__':
    parser = utils.make_basic_parser()
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)

    if not args.use_large_feats:
        raise ValueError('Use large feats!')

    # check if have to run
    save_directory = f'./outputs/{utils.get_dataset_name_from_datapath(args.data_dir)}_{utils.get_model_basename(args.model_name)}'
    if args.ds_name is None:
        name = args.datasplit
    else:
        name = f'{args.ds_name}_{args.datasplit}'
    folder_name = f'{save_directory}/generations/ref'


    device = utils.get_device_from_arg(args.device)
    print(f'Using device: {device}')


    ###### OLD
    ## featurize samples
    # feats = src.model_utils.featurize_sequential(model, ds_tokens)
    # torch.save(feats, f'{folder_name}/feats_{name}.pt')


    feats_prefix = ''
    if args.use_large_feats:
        model, tokenizer = utils.get_model_and_tokenizer(model_name=args.featurize_model_name, device=device)
        ds_tokens = utils.load_and_tokenize_data(tokenizer, args.data_dir, args.max_len, args.max_num_generations,
                                                 ds_name=args.ds_name, split=args.datasplit)
        for l in {128, 256, 512, args.max_len}:
            feats_prefix = f'L{l}'
            feats_out_fn = f'{folder_name}/feats{feats_prefix}_{name}.pt'
            if os.path.isfile(feats_out_fn):
                print(f'Feats {feats_out_fn} exisits. Skipping')
                continue
            else:
                print(f'Featurizing l = {l}...')
                samples_3 = [x[:, :l] for x in ds_tokens]
                feats = src.model_utils.featurize_sequential(model, samples_3)
                torch.save(feats, feats_out_fn)
    else:  # use features from model
        model, tokenizer = utils.get_model_and_tokenizer(model_name=args.model_name, device=device)
        ds_tokens = utils.load_and_tokenize_data(tokenizer, args.data_dir,
                                                 args.max_len, args.max_num_generations, split=args.datasplit)
        feats = src.model_utils.featurize_sequential(model, ds_tokens)
        torch.save(feats, f'{folder_name}/feats_{name}.pt')

