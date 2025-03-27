from nn_models.models.pure_conv import CustomConv1D
from nn_models.models.pure_lstm import CustomLSTM, rot6_to_rotmat
from _2_optimization.utils.optimization_utils import optimization_demo

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
    Zero-lag Butterworth filter for data along axis=1.
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

    def print_result(label, arr):
        # If arr is shape (3,), unpack as flex/add/rot
        if isinstance(arr, np.ndarray) and arr.size == 3:
            flex, add, rot = arr
            print('%s %.2f (Flexion),  %.2f (Adduction),  %.2f (Rotation)' % (label, flex, add, rot))
        else:
            # Otherwise, treat it as a single average RMSE
            val = float(arr) if isinstance(arr, np.ndarray) else arr
            print('%s %.2f (Avg RMSE)' % (label, val))

    # Optionally save results
    import os
    if result_fldr is not None:
        if not os.path.exists(result_fldr):
            os.makedirs(result_fldr)
        calib_name = "calib_" if calib else ""
        np.save(os.path.join(result_fldr, calib_name + "nn_result.npy"), nn_result)
        np.save(os.path.join(result_fldr, calib_name + "combined_result.npy"), combined_result)

    # If we have ground-truth, compute per-angle or single average
    if gt_angle is not None:
        if calib:
            gt_angle       = gt_angle - gt_angle.mean(axis=1)[:, None, :]
            nn_result      = nn_result - nn_result.mean(axis=1)[:, None, :]
            combined_result= combined_result - combined_result.mean(axis=1)[:, None, :]

        rmse_nn  = np.sqrt(((nn_result - gt_angle)**2).mean(axis=1)).mean(axis=0)
        rmse_opt = np.sqrt(((combined_result - gt_angle)**2).mean(axis=1)).mean(axis=0)
    else:
        rmse_nn  = np.nan
        rmse_opt = np.nan

    c_str = '(Calibrated)' if calib else '(Uncalibrated)'
    print_result('Neural Network %s  :' % c_str, rmse_nn)
    print_result('Optimization %s    :' % c_str, rmse_opt)
    return rmse_nn, rmse_opt


def run_demo(inpt_data, gyro_data,
             angle_norm_dict, ori_norm_dict,
             angle_model, ori_model,
             weight, std_ratio, result_fldr,
             joint='Knee', leg='Left', gt_angle=None,
             **kwargs):
    """
    1) Normalize data
    2) Pass into angle/orientation models
    3) Zero-pad orientation output to 60 if needed
    4) Perform optimization
    5) Evaluate RMSE
    """

    ## TODO - need to chagne original code is running from 0
    # For demonstration, keep only batch=1
    inpt_data = inpt_data[:1]   # shape => [1, frames, dims]
    gyro_data = gyro_data[:1]   # shape => [1, frames, dims]
    if gt_angle is not None:
        gt_angle = gt_angle[:1]

    with torch.no_grad():
        # Normalize inputs for angle & orientation
        inpt_data_angle = (inpt_data - angle_norm_dict['x_mean']) / angle_norm_dict['x_std']
        inpt_data_ori   = (inpt_data - ori_norm_dict['x_mean'])   / ori_norm_dict['x_std']

        angle_model.eval()
        alpha = angle_model(inpt_data_angle)

        ori_model.eval()
        ori_pred = ori_model(inpt_data_ori)

        ## TODO - just skipped for running
        # Example: if we needed rot6 => rot6_to_rotmat, do it here
        # ori_pred = rot6_to_rotmat(ori_pred)

        # "Un-normalize" angles
        alpha = alpha * angle_norm_dict['y_std'] + angle_norm_dict['y_mean']

        alpha    = alpha.detach().cpu().double().numpy()    # shape [1, frames, angle_dims]
        ori_pred = ori_pred.detach().cpu().double().numpy() # shape [1, frames, ori_dims]

    ## TODO - need to delete later
    # >>>>>> If your pipeline expects 60 channels in ori_pred, zero-pad it here <<<<<<
    needed_dims = 60
    current_dims = ori_pred.shape[-1]
    if current_dims < needed_dims:
        pad_size = needed_dims - current_dims
        pad_zeros = np.zeros((ori_pred.shape[0], ori_pred.shape[1], pad_size))
        ori_pred = np.concatenate([ori_pred, pad_zeros], axis=-1)
        print(f"[INFO] Zero-padded ori_pred from {current_dims} -> {needed_dims} channels")

    ## TODO - need to check this part again 
    # Now ori_pred has last-dim=60 => np.split(..., 2, -1) => 2x30
    beta = optimization_demo(ori_pred, gyro_data, joint=joint, leg=leg)

    # Combine alpha & beta
    beta = (beta - beta.mean(axis=1)[:, None]) * std_ratio + alpha.mean(axis=1)[:, None]
    theta= weight * alpha + (1 - weight) * beta

    rmse_nn, rmse_opt = evaluate_result(alpha, theta, gt_angle, result_fldr=result_fldr, calib=False)
    print('\n')
    rmse_nn_calib, rmse_opt_calib = evaluate_result(alpha, theta, gt_angle, result_fldr=result_fldr, calib=True)
    save_to_csv(result_fldr, rmse_nn, rmse_opt, rmse_nn_calib, rmse_opt_calib)

def load_custom_data(path, is_imu_data=True):
    """
    Loads IMU data from .npy or .pkl. If 2D => expand to 3D. 
    If dims not in [3,4], skip norm augmentation.
    """
    import torch, pickle, numpy as np
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
        print(f"[INFO] Loaded data from {path}, dtype: {arr.dtype}, shape: {arr.shape}")

        if arr.dtype == np.object_:
            print(f"[WARNING] Converting object array to float64 for {path}")
            arr = np.array([np.array(item, dtype=np.float64) for item in arr])
            print(f"[INFO] After conversion: dtype: {arr.dtype}, shape: {arr.shape}")

        # Example: fill with dummy shape if needed
        ## TODO need to delete
        arr = np.ones((3600, 3))

        if arr.ndim == 2:
            print("[INFO] Data is 2D, expanding to 3D by adding a leading dimension.")
            arr = np.expand_dims(arr, axis=0)

        arr = torch.from_numpy(arr)

    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            arr = pickle.load(f)
            if isinstance(arr, np.ndarray):
                arr = torch.from_numpy(arr)
            else:
                assert isinstance(arr, torch.Tensor), "Unsupported data type"
    else:
        raise NotImplementedError(f"Unsupported format {path[-3:]}")

    if not is_imu_data:
        if isinstance(arr, torch.Tensor):
            arr = arr.double().numpy()
        return arr

    b, l, d = arr.shape
    if d not in [3, 4]:
        print(f"[INFO] IMU data dimension ({d}) is not 3 or 4. Skipping norm augmentation.")
    else:
        if d == 3:
            norm = torch.norm(arr, p='fro', dim=-1, keepdim=True)
            arr  = torch.cat([arr, norm], dim=-1)

    return arr

def prepare_data(root_path, leg, device, dtype):
    """
    Gathers 4 .npy files => merges => shape [batch, frames, dims].
    Also zero-pads inpt_data -> 60 channels if needed, in case we want 2x30 split.
    """
    ## TODO - Need to change
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

    ## TODO -- just add # Zero-pad merged input to 60 channels if needed
    dims_needed = 60
    curr_dims   = inpt_data.shape[-1]
    if curr_dims < dims_needed:
        pad_amount  = dims_needed - curr_dims
        pad_tensor  = torch.zeros(
            inpt_data.shape[0],
            inpt_data.shape[1],
            pad_amount,
            device=inpt_data.device,
            dtype=inpt_data.dtype
        )
        inpt_data = torch.cat([inpt_data, pad_tensor], dim=-1)

        # likewise for inpt_gyro
        gyr_dims = inpt_gyro.shape[-1]
        if gyr_dims < dims_needed:
            pad_np = np.zeros((inpt_gyro.shape[0], inpt_gyro.shape[1], dims_needed - gyr_dims))
            inpt_gyro = np.concatenate([inpt_gyro, pad_np], axis=-1)

    return inpt_data, inpt_gyro, gt_angle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo code arguments')

    parser.add_argument('--joint', choices=["Knee", "Hip", "Ankle"], type=str)
    parser.add_argument('--activity', choices=["Walking", "Running"], type=str)
    parser.add_argument('--root-path', type=str)
    parser.add_argument('--angle-model-fldr', type=str, default="")
    parser.add_argument('--ori-model-fldr', type=str, default="")
    parser.add_argument('--result-fldr', type=str, default="")
    parser.add_argument('--use-cuda', default=True,
                        type=lambda x: x.lower() in ['true', '1'])

    args = parser.parse_args()

    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    dtype  = torch.float

    joint    = args.joint
    activity = args.activity
    root_path= osp.join(args.root_path, joint)
    leg      = 'Left'

    # 1) gather data
    inpt_data, inpt_gyro, gt_angle = prepare_data(root_path, leg, device, dtype)

    # 2) dummy angle/orientation models
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # For example, produce shape [batch, frames, 3]
            return torch.zeros(x.shape[0], x.shape[1], 3, device=x.device, dtype=x.dtype)

    ## TODO - need to change to real model
    angle_model = DummyModel()
    ori_model   = DummyModel()

    # 3) minimal norm dict
    angle_norm_dict = {
        'x_mean': torch.zeros_like(inpt_data[0]),
        'x_std':  torch.ones_like(inpt_data[0]),
        'y_mean': 0.0,
        'y_std':  1.0
    }
    # Suppose orientation dict is identical
    ori_norm_dict = angle_norm_dict.copy()

    # 4) default optimization parameters
    weight    = 0.5
    std_ratio = 1.0

    # 5) run
    run_demo(inpt_data, inpt_gyro,
             angle_norm_dict, ori_norm_dict,
             angle_model, ori_model,
             weight, std_ratio,
             args.result_fldr,
             joint=joint, leg=leg, gt_angle=gt_angle)
