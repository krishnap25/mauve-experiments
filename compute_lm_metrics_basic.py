import argparse
import os, time, pickle as pkl

import src.utils as utils
import src.metrics

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=-1,
                        help='choose one of [0, 1, 2, 3] for GPU, or CPU otherwise')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--datasplit', type=str, default='valid')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--max_len', type=int, default=1024)
    parser.add_argument('--max_num_data', type=int, default=5000)
    return parser

def get_metrics(param, metric_fn_lst, model, ds_tokens, datasplit, metric_fn_names, save_directory):
    # param = (top_, top_k, temp)
    p, k, temp = param
    output_file_name = f'{save_directory}/metrics/basic/lm_{datasplit}_p{p}_k{k}_t{temp}.p'
    if os.path.isfile(output_file_name):
        print(f'{output_file_name} existing. Exiting.')
        return
    t1 = time.time()
    metrics = src.metrics.compute_metrics_from_probs(
        model, ds_tokens, metric_fn_lst, eppl_eps_lst=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0],
        temperature=temp, top_k=k, top_p=p,
    )
    t2 = time.time()
    print(metrics, round(t2-t1, 2))

    with open(output_file_name, 'wb') as f:
        pkl.dump([metrics, metric_fn_names], f)

def main():
    parser = make_parser()
    args = parser.parse_args()
    print(args)

    device = utils.get_device_from_arg(args.device)
    print(f'Using device: {device}')

    model, tokenizer = utils.get_model_and_tokenizer(model_name=args.model_name, device=device)
    save_directory = f'./outputs/{utils.get_dataset_name_from_datapath(args.data_dir)}_{utils.get_model_basename(args.model_name)}'

    ds_tokens = utils.load_and_tokenize_data(tokenizer, args.data_dir,
                                             args.max_len, args.max_num_data, split=args.datasplit)

    metric_fn_lst = src.metrics.get_probs_metric_fn_lst()
    metric_fn_names = src.metrics.get_metric_names()
    print(metric_fn_names)

    for p in [0.8, 0.9, 0.92, 0.95, 0.99]: # 5
        param = (p, 0, 1.0)
        get_metrics(param, metric_fn_lst, model, ds_tokens, args.datasplit, metric_fn_names, save_directory)

    for k in [1, 5, 10, 50, 100, 500, 1000, 2000, 5000, 10000]: # 10
        param = (1.0, k, 1.0)
        get_metrics(param, metric_fn_lst, model, ds_tokens, args.datasplit, metric_fn_names, save_directory)

    for t in [0.7, 0.8, 0.9, 0.95, 1.0]: # 5
        param = (1.0, 0, t)
        get_metrics(param, metric_fn_lst, model, ds_tokens, args.datasplit, metric_fn_names, save_directory)

    for t in [0.75, 0.9]: # 4
        for k in [10, 100]:
            param = (1.0, k, t)
        get_metrics(param, metric_fn_lst, model, ds_tokens, args.datasplit, metric_fn_names, save_directory)


if __name__ == '__main__':
    main()
