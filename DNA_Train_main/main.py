import argparse
import collections
import logging
import os
import random as rn
import time

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn

import data_loader
from Streaming_model import StreamingVC
from utils import Make_file, cal_std


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


I_D = {
    0: "oxford4",
    1: "ALOI_100",
    2: "NUS",
    3: "OutdoorScene",
    4: "ALOI_10",
    5: "YoutubeFace_sel_fea",
}

METRIC_COLUMNS = ["acc", "nmi", "ari"]


def set_seed(seed):
    np.random.seed(seed)
    rn.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = False


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_one_alpha(config):
    seeds = [1, 2, 3, 4, 5]#seeds = [1, 2, 3, 4, 5]
    path, filename = Make_file(config)
    loss_para = [config.beta1, config.beta2]
    accumulated_metrics = collections.defaultdict(list)

    for seed in seeds:
        logging.info("--------alpha:{:.1f}, seed:{}--------".format(config.similarity_alpha, seed))
        set_seed(seed)

        print("Information", config)
        print("similarity_alpha: {:.1f}".format(config.similarity_alpha))
        print("seed: {}".format(seed))

        dsvc = StreamingVC(config)
        acc, nmi, ari = dsvc.train(path, filename, loss_para)
        accumulated_metrics["acc"].append(acc)
        accumulated_metrics["nmi"].append(nmi)
        accumulated_metrics["ari"].append(ari)

    metrics_summary = cal_std(
        accumulated_metrics["acc"],
        accumulated_metrics["nmi"],
        accumulated_metrics["ari"],
    )

    summary_row = {"similarity_alpha": config.similarity_alpha}
    result_row = [config.similarity_alpha]
    for metric_name, mean, std in zip(METRIC_COLUMNS, metrics_summary[::2], metrics_summary[1::2]):
        summary_row[f"{metric_name}_mean"] = mean
        summary_row[f"{metric_name}_std"] = std
        result_row.append(f"{mean:.2f} / {std:.2f}")

    print(result_row)
    pd.DataFrame([result_row]).to_csv(path + filename, mode="a", header=False, index=False)
    return summary_row


def build_base_config(dataset_id, dataset_name, data_para, args):
    config = argparse.Namespace()
    config.i_d = dataset_id
    config.datasets = data_para
    config.compute_all = False
    config.mode = "train"
    config.running_time = False
    config.cuda_list = [0]
    config.test_time = 5
    config.lr = args.lr
    config.beta1 = data_para["beta1"]
    config.beta2 = data_para["beta2"]
    config.similarity_alpha = data_para["similarity_alpha"]
    config.batch_size = args.batch_size
    config.feature_dim = args.feature_dim
    config.Knowledge_size = data_para["Knowledge_size"]
    config.view_num = data_para["V"]
    config.Cluster = data_para["K"]
    config.sample_interval = 1
    config.epochs = args.epochs
    config.dataset_name = dataset_name
    return config


def save_alpha_summary(dataset_name, rows):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_dir = os.path.join(base_dir, "logs", dataset_name)
    os.makedirs(summary_dir, exist_ok=True)
    current_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime(time.time()))
    summary_file = os.path.join(summary_dir, f"{dataset_name}_{current_time}.csv")
    pd.DataFrame(rows).to_csv(summary_file, index=False)
    return summary_file


def run_experiments(args):
    dataset_ids = parse_int_list(args.dataset_ids)
    all_data = data_loader.ALL_data

    for dataset_id in dataset_ids:
        dataset_name = I_D[dataset_id]
        print(f"Processing dataset: {dataset_name}")
        data_para = all_data[dataset_name]
        config = build_base_config(dataset_id, dataset_name, data_para, args)
        summary_rows = [run_one_alpha(config)]

        summary_file = save_alpha_summary(dataset_name, summary_rows)
        print(f"Alpha summary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_ids", type=str, default="0", help="Comma-separated dataset ids. Default: 5 (YoutubeFace_sel_fea).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    run_experiments(parser.parse_args())
