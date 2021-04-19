import os
import pickle as pkl
import torch

import src.utils as utils
import src.mauve_metrics as mauve_metrics


def main():
    parser = utils.make_metrics_parser()
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    device = utils.get_device_from_arg(args.device)
    save_directory = f'./outputs/{utils.get_dataset_name_from_datapath(args.data_dir)}_{utils.get_model_basename(args.model_name)}'

    if args.use_large_feats:
        feats_suffix = f'L{args.max_len}'
    elif args.use_bert_feats:
        feats_suffix = f'B{args.max_len}'
    else:
        feats_suffix = ''

    if args.use_large_feats:
        print('---------------Using features from GPT-2 Large!!!! Suffix =', feats_suffix)
    elif args.use_bert_feats:
        print('---------------Using features from Roberta Large!!!! Suffix =', feats_suffix)
    else:
        print('---------------Using features from model used for generations!!!!')

    if not os.path.isfile(f'{save_directory}/generations/ref/feats{feats_suffix}_{args.datasplit}.pt'):
        raise FileNotFoundError(f'Generations {save_directory}/generations/ref/feats{feats_suffix}_{args.datasplit}.pt do not exist')
    p_feats = torch.load(f'{save_directory}/generations/ref/feats{feats_suffix}_{args.datasplit}.pt')
    folder, filename = utils.get_save_filename_from_args(args)

    algo_name = mauve_metrics.get_discretization_algo_name(
        discretization_algo=args.discretization,
        kmeans_num_clusters=args.kmeans_num_clusters, kmeans_explained_var=args.kmeans_explained_var,
        drmm_num_epochs=args.drmm_num_epochs, drmm_n_layer=args.drmm_n_layer,
        drmm_n_comp_per_layer=args.drmm_n_component_per_layer,
        spv_num_epochs=args.spv_num_epochs, device=device, seed=args.seed+1
    )
    savefilename = f'{save_directory}/metrics/{folder}/mauve_{feats_suffix}_{filename}_{algo_name}.p'
    if os.path.isfile(savefilename) and not args.force:
        print('Metrics already exist. Exiting')
        return

    if not os.path.isfile(f'{save_directory}/generations/{folder}/feats{feats_suffix}_{filename}.pt'):
        raise FileNotFoundError(f'Generations {save_directory}/generations/{folder}/feats{feats_suffix}_{filename}.pt do not exist')

    q_feats = torch.load(f'{save_directory}/generations/{folder}/feats{feats_suffix}_{filename}.pt')

    p_quant, q_quant, metrics = mauve_metrics.compute_mauve_metrics(
        p_feats, q_feats, discretization_algo=args.discretization,
        kmeans_num_clusters=args.kmeans_num_clusters, kmeans_explained_var=args.kmeans_explained_var,
        drmm_num_epochs=args.drmm_num_epochs, drmm_n_layer=args.drmm_n_layer,
        drmm_n_comp_per_layer=args.drmm_n_component_per_layer,
        spv_num_epochs=args.spv_num_epochs, device=device, seed=args.seed+1
    )
    print('Mauve metric:', metrics)

    # save
    with open(savefilename, 'wb') as f:
        pkl.dump([metrics, p_quant, q_quant], f)
    print(f'Done. Saved "{savefilename}". Bye!')


if __name__ == '__main__':
    main()
