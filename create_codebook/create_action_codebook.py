import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import json
import glob
from pathlib import Path

import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.token_process import TokenProcessor
from utils.rollout import cal_polygon_contour, transform_to_local, wrap_angle
import argparse


def Kdisk_cluster(
    X,  # [n_trajs, 4, 2], bbox of the last point of the segment
    N,  # int
    tol,  # float
    a_pos,  # [n_trajs, 6, 3], the complete segment
    cal_mean_heading=True,
):
    n_total = X.shape[0]
    ret_traj_list = []

    for i in range(N):
        if i == 0:
            choice_index = 0  # always include [0, 0, 0]
        else:
            choice_index = torch.randint(0, X.shape[0], (1,)).item()

        x0 = X[choice_index]

        res_mask = torch.norm(X - x0, dim=-1).mean(-1) > tol
        if cal_mean_heading:
            ret_traj = a_pos[~res_mask].mean(0, keepdim=True)
        else:
            ret_traj = a_pos[[choice_index]]

        X = X[res_mask]
        a_pos = a_pos[res_mask]
        ret_traj_list.append(ret_traj)

        remain = X.shape[0] * 100.0 / n_total
        n_inside = (~res_mask).sum().item()
        print(f"{i=}, {remain=:.2f}%, {n_inside=}")
        
    return torch.cat(ret_traj_list, dim=0)  # [N, 6, 3]


if __name__ == "__main__":
    # set seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="/data/temp/Expire180Days/users/sethzhao/carla_data/cached_Bench2Drive_train_fixedheading_autovla_cleaned")
    parser.add_argument("--out_file_name", type=str, default="/data/temp/Expire180Days/users/sethzhao/planning-intern/agent_vocab_mixed_waymo_nuplan_kdisk.pkl")
    parser.add_argument("--vocab_size", type=int, default=2048)
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # data
    n_trajs = 2048 * 200
    data_paths = [
        "/data/temp/Expire180Days/users/sethzhao/navsim_data/nuplan_trainval_autovla_cleaned",
        # "/data/temp/Expire180Days/users/sethzhao/carla_data/cached_carlagarage_Apr30_v2",
        "/data/temp/Expire180Days/users/sethzhao/WaymoE2E/training_cot_07222025",
        # "/data/temp/Expire180Days/users/sethzhao/carla_data/cached_Bench2Drive_train_fixedheading_autovla_cleaned"
    ]
    out_file_name = args.out_file_name
    tol_dist = 0.05
    num_cluster = args.vocab_size

    trajs = torch.zeros([1, 1, 3], dtype=torch.float32) # ego veh

    for data_path in data_paths:
        data_file_path = glob.glob(f"{data_path}/*.json")
        count = 0
        with tqdm(total=len(data_file_path), desc=f"n_trajs={n_trajs}") as pbar:
            for file in data_file_path:
                with open(file, 'r') as f:
                    data = json.load(f)
                trajectory = data['gt_trajectory']
                pos = torch.tensor([[0, 0]])
                head = torch.tensor([0])
                for t in range(len(trajectory)):
                    if count < n_trajs:
                        next_pos = torch.tensor([trajectory[t][:2]])
                        next_head = torch.tensor([trajectory[t][2]])
                        l_pos, l_head = transform_to_local(
                            pos_global=next_pos.unsqueeze(0),  # [1, 1, 2]
                            head_global=next_head.unsqueeze(0),  # [1, 1]
                            pos_now=pos,  # [1, 2]
                            head_now=head,  # [1]
                        )
                        # print(l_pos)
                        l_head = wrap_angle(l_head)
                        to_add = torch.cat([l_pos, l_head.unsqueeze(-1)], dim=-1)
                        #if not ((trajs - to_add).abs().sum([1, 2]) < 1e-2).any()
                        trajs = torch.cat([trajs, to_add], dim=0)
                        count += 1
                        pos = next_pos.clone()
                        head = next_head.clone()
                pbar.update(1)
                if count >= n_trajs:
                    break
        #     print(trajs)

    # trajs = torch.stack(trajs, dim=0).squeeze(dim=2)
    print(trajs.shape)
    # exit()
    res = {"token_all": {}}

    # K-disk cluster
    width_length = torch.tensor([2.0, 4.8])
    width_length = width_length.unsqueeze(0)  # [1, 2]

    contour = cal_polygon_contour(
        pos=trajs[:, -1, :2], head=trajs[:, -1, 2], width_length=width_length
    )  # [n_trajs, 4, 2]

    ret_traj = Kdisk_cluster(X=contour, N=num_cluster, tol=tol_dist, a_pos=trajs)
    ret_traj[:, :, -1] = wrap_angle(ret_traj[:, :, -1])

    plt.scatter(ret_traj[:, 0, 0], ret_traj[:, 0, 1])
    plt.axis('equal')
    plt.savefig('nuplan_token.jpg')

    contour = cal_polygon_contour(
        pos=ret_traj[:, :, :2],  # [N, 6, 2]
        head=ret_traj[:, :, 2],  # [N, 6]
        width_length=width_length.unsqueeze(0),
    )

    res["token_all"]["veh"] = contour.numpy()

    with open(Path(__file__).resolve().parent / out_file_name, "wb") as f:
        pickle.dump(res, f)