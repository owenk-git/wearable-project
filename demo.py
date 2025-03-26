from nn_models.models.pure_conv import CustomConv1D
from nn_models.models.pure_lstm import CustomLSTM, rot6_to_rotmat
from _2_optimization.utils.optimization_utils import *

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

import pickle
import argparse
import os
from os import path as osp


def butter_low(data, order=4, fc=5, fs=100):
    """
    Zero-lag Butterworth filter for column data (axis=1).
    """
    b, a = butter(order, 2 * fc / fs, 'low')
    padlen = min(int(0.5 * data.shape[0]), 200)
    return filtfilt(b, a, data, padlen=padlen, axis=1)


def save_to_csv(path, nn, opt, calib_nn, calib_opt):
    """
    Creates a CSV file summarizing RMSE results.
    """
    import pandas as pd
    if path is not None:
        cols = ['RMSE Type', 'Flexion', 'Adduction', 'Rotation']
        df_one = pd.DataFrame(columns=cols)
        df_one.loc[0] = ['NN'] + nn.tolist()
        df_one.loc[1] = ['NN + Opt'] + opt.tolist()
        df_one.loc[2] = ['Calibrated NN'] + calib_nn.to_list()
        df_one.loc[3] = ['Calibrated NN + Opt'] + calib_opt.to_list()
        df_one.to_csv(path + '/RMSE_loss.csv', index=False)


def evaluate_result(nn_result, combined_result, gt_angle, result_fldr=None, calib=False):
    """
    Calculate RMSE results (Neural Network vs. Optimization) by comparing with ground truth angles.
    """

    def print_result(label, array):
        flex, add, rot = array
        print('%s %.2f (Flexion),  %.2f (Adduction),  %.2f (Rotation)' % (label, flex, add, rot))

    if result_fldr is not None:
        if not osp.exists(result_fldr):
            os.makedirs(result_fldr)
        calib_name = "calib_" if calib else ""
        np.save(osp.join(result_fldr, calib_name + "nn_result.npy"), nn_result)
        np.save(osp.join(result_fldr, calib_name + "combined_result.npy"), combined_result)

    if gt_angle is not None:
        # Optionally remove mean if calibrating
        gt_angle = gt_angle - gt_angle.mean(axis=1)[:, None, :] if calib else gt_angle
        nn_result = nn_result - nn_result.mean(axis=1)[:, None, :] if calib else nn_result
        combined_result = combined_result - combined_result.mean(axis=1)[:, None, :] if calib else combined_result

        rmse_nn_result = np.sqrt(((nn_result - gt_angle) ** 2).mean(axis=1)).mean(axis=0)
        rmse_opt_result = np.sqrt(((combined_result - gt_angle) ** 2).mean(axis=1)).mean(axis=0)
    else:
        rmse_nn_result = np.nan
        rmse_opt_result = np.nan

    c_str = '(Calibrated)' if calib else '(Uncalibrated)'
    print_result('Neural Network %s  :' % c_str, rmse_nn_result)
    print_result('Optimization %s    :' % c_str, rmse_opt_result)

    return rmse_nn_result, rmse_opt_result


def run_demo(inpt_data, gyro_data,
             angle_norm_dict, ori_norm_dict,
             angle_model, ori_model,
             weight, std_ratio, result_fldr,
             joint='Knee', leg='Left', gt_angle=None,
             **kwargs):
    """
    Runs the demo: 
    1) Normalizes data 
    2) Feeds it into angle & orientation models 
    3) Optionally performs optimization 
    4) Evaluates RMSE 
    """

    # We'll only keep the batch dimension slice for demonstration: inpt_data[:1].
    inpt_data = inpt_data[:1]  # shape [1, 417, dims]
    gyro_data = gyro_data[:1]  # shape [1, 417, dims]

    if gt_angle is not None:
        gt_angle = gt_angle[:1]

    with torch.no_grad():
        # Normalize
        inpt_data_angle = (inpt_data - angle_norm_dict['x_mean']) / angle_norm_dict['x_std']
        inpt_data_ori   = (inpt_data - ori_norm_dict['x_mean']) / ori_norm_dict['x_std']

        angle_model.eval()
        alpha = angle_model(inpt_data_angle)

        ori_model.eval()
        ori_pred = ori_model(inpt_data_ori)
        # ori_pred = rot6_to_rotmat(ori_pred)

        # un-normalize angle
        alpha = alpha * angle_norm_dict['y_std'] + angle_norm_dict['y_mean']

        alpha = alpha.detach().cpu().double().numpy()
        ori_pred = ori_pred.detach().cpu().double().numpy()

    # Optimization step
    beta = optimization_demo(ori_pred, gyro_data, joint=joint, leg=leg)
    beta = (beta - beta.mean(axis=1)[:, None]) * std_ratio + alpha.mean(axis=1)[:, None]
    # We keep all frames, shape is [1, 417, ...]
    theta = weight * alpha + (1 - weight) * beta

    rmse_nn, rmse_opt = evaluate_result(alpha, theta, gt_angle, result_fldr=result_fldr, calib=False)
    print('\n\n')
    rmse_nn_calib, rmse_opt_calib = evaluate_result(alpha, theta, gt_angle, result_fldr=result_fldr, calib=True)
    save_to_csv(result_fldr, rmse_nn, rmse_opt, rmse_nn_calib, rmse_opt_calib)


def load_custom_data(path, is_imu_data=True):
    """
    Load IMU data from a path. If the data is not shaped (batch, length, dims) = (B, L, 3 or 4),
    we skip norm augmentation. If 2D, expand to 3D by adding batch dimension.
    """
    import torch, pickle, numpy as np
    from os import path as osp

    if path.endswith(".npy"):
        _data = np.load(path, allow_pickle=True)
        print(f"[INFO] Loaded data from {path}, dtype: {_data.dtype}, shape: {_data.shape}")

        if _data.dtype == np.object_:
            print(f"[WARNING] Converting object array to float64 for {path}")
            _data = np.array([np.array(item, dtype=np.float64) for item in _data])
            print(f"[INFO] After conversion: dtype: {_data.dtype}, shape: {_data.shape}")

        if _data.ndim == 2:
            print("[INFO] Data is 2D, expanding to 3D by adding a leading dimension.")
            _data = np.expand_dims(_data, axis=0)

        _data = torch.from_numpy(_data)

    elif path.endswith(".pkl"):
        with open(path, "rb") as fopen:
            _data = pickle.load(fopen)
            if isinstance(_data, np.ndarray):
                _data = torch.from_numpy(_data)
            else:
                err_msg = f"Data type {type(_data)} is not supported"
                assert isinstance(_data, torch.Tensor), err_msg
    else:
        err_msg = f"Input file format {path[-3:]} is not supported"
        raise NotImplementedError(err_msg)

    if not is_imu_data:
        if isinstance(_data, torch.Tensor):
            _data = _data.double().numpy()
        return _data

    sz_b, sz_l, sz_d = _data.shape
    if sz_d not in [3, 4]:
        print(f"[INFO] IMU data dimension ({sz_d}) is not 3 or 4. Skipping norm augmentation.")
    else:
        if sz_d == 3:
            norm = torch.norm(_data, p='fro', dim=-1, keepdim=True)
            _data = torch.cat([_data, norm], dim=-1)

    return _data


def prepare_data(root_path, leg, device, dtype):
    """
    Gather seg1_acc, seg2_acc, seg1_gyro, seg2_gyro from user files 
    and combine them into a single input tensor.
    """

    seg1_accel_path = osp.join(root_path, f'{leg}_seg1_acc.npy')
    seg2_accel_path = osp.join(root_path, f'{leg}_seg1_acc.npy')
    seg1_gyro_path  = osp.join(root_path, f'{leg}_seg1_acc.npy')
    seg2_gyro_path  = osp.join(root_path, f'{leg}_seg1_acc.npy')

    seg1_accel = load_custom_data(seg1_accel_path)
    seg2_accel = load_custom_data(seg2_accel_path)
    seg1_gyro  = load_custom_data(seg1_gyro_path)
    seg2_gyro  = load_custom_data(seg2_gyro_path)

    gt_angle = None
    inpt_data = torch.cat([seg1_accel, seg1_gyro, seg2_accel, seg2_gyro], dim=-1).to(device=device, dtype=dtype)
    inpt_gyro = torch.cat([seg1_gyro[:, :, :-1], seg2_gyro[:, :, :-1]], dim=-1).double().numpy()
    return inpt_data, inpt_gyro, gt_angle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo code arguments')

    parser.add_argument('--joint', choices=["Knee", "Hip", "Ankle"],
                        type=str, help="Joint (Knee/Hip/Ankle)")
    parser.add_argument('--activity', choices=["Walking", "Running"],
                        type=str, help="Activity (Walking/Running)")
    parser.add_argument('--root-path', type=str, help="Root path to data")
    parser.add_argument('--angle-model-fldr', type=str, default="",
                        help="Folder path for angle model")
    parser.add_argument('--ori-model-fldr', type=str, default="",
                        help="Folder path for orientation model")
    parser.add_argument('--result-fldr', type=str, default="",
                        help="Output folder for results")
    parser.add_argument('--use-cuda', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use GPU if available')

    args = parser.parse_args()

    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    dtype = torch.float

    joint = args.joint
    activity = args.activity
    root_path = osp.join(args.root_path, joint)
    leg = 'Left'

    inpt_data, inpt_gyro, gt_angle = prepare_data(root_path, leg, device, dtype)

    # Suppose we have placeholders:
    angle_model = None  
    ori_model   = None

    # placeholders for norm dictionaries
    angle_norm_dict = {
        'x_mean': torch.zeros_like(inpt_data[0]),
        'x_std':  torch.ones_like(inpt_data[0]),
        'y_mean': 0.0,
        'y_std':  1.0
    }
    ori_norm_dict = angle_norm_dict

    if angle_model is None:
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                # e.g. just return zero
                return torch.zeros(x.shape[0], x.shape[1], 3, device=x.device, dtype=x.dtype)
        angle_model = DummyModel()
        ori_model   = DummyModel()

    # default optimization parameters
    weight = 0.5
    std_ratio = 1.0

    run_demo(inpt_data, inpt_gyro,
             angle_norm_dict, ori_norm_dict,
             angle_model, ori_model,
             weight, std_ratio,
             args.result_fldr,
             joint=joint, leg=leg, gt_angle=gt_angle)
