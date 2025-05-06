from nn_models.models.pure_conv import CustomConv1D
from nn_models.models.pure_lstm import CustomLSTM, rot6_to_rotmat
from nn_models.models.pure_transformer import transformer
from _2_optimization.utils.optimization_utils import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks

import pickle
import argparse
import os
from os import path as osp
from scipy.spatial.transform import Rotation as R

import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_imu_data(data, x_rot=None, y_rot=None, z_rot=None, custom_matrix=None):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    if custom_matrix is not None:
        T = torch.tensor(custom_matrix.T, dtype=data.dtype, device=data.device)
    else:
        r_z = R.from_euler('z', z_rot, degrees=True)
        r_x = R.from_euler('x', x_rot, degrees=True)
        r_y = R.from_euler('y', y_rot, degrees=True)
        T = torch.tensor((r_y.as_matrix() @ r_x.as_matrix() @ r_z.as_matrix()).T, dtype=data.dtype)

    rotated = data[..., :3] @ T
    norm = torch.norm(rotated, p='fro', dim=-1, keepdim=True)
    return torch.cat([rotated, norm], dim=-1)


def save_predictions_to_csv_separate(folder_path, nn_result, opt_result, gt_angle):
    """
    Save NN, Opt, and Ground Truth angle predictions into separate CSV files.

    Parameters:
        folder_path (str): directory to save CSVs
        nn_result (np.ndarray): shape (1, T, 3)
        opt_result (np.ndarray): shape (1, T, 3)
        gt_angle (np.ndarray): shape (1, T, 3)
    """

    os.makedirs(folder_path, exist_ok=True)
    T = nn_result.shape[1]

    def create_df(data, label):
        return pd.DataFrame({
            "Time": range(T),
            "Flexion": data[0, :, 0],
            "Adduction": data[0, :, 1],
            "Rotation": data[0, :, 2]
        })

    df_nn = create_df(nn_result, "NN")
    df_opt = create_df(opt_result, "NN + Opt")
    df_gt = create_df(gt_angle, "GroundTruth")

    df_nn.to_csv(os.path.join(folder_path, "nn_prediction.csv"), index=False)
    df_opt.to_csv(os.path.join(folder_path, "opt_prediction.csv"), index=False)
    df_gt.to_csv(os.path.join(folder_path, "ground_truth.csv"), index=False)



def find_alignment_peaks(signal_1, signal_2, prominence_1=0.2, prominence_2=0.2, distance=50, normalize=True, plot=True):
    """
    Find the first 3 peaks in two signals and calculate the alignment offset.
    
    Parameters:
    - signal_1: np.array, first signal (e.g., IMU)
    - signal_2: np.array, second signal (e.g., angle)
    - height_1: float, peak height threshold for signal_1
    - height_2: float, peak height threshold for signal_2
    - distance: int, minimum distance between peaks
    - plot: bool, whether to show plots with detected peaks

    Returns:
    - imu_peaks_3: first 3 peak indices in signal_1
    - angle_peaks_3: first 3 peak indices in signal_2
    - time_shift: offset in indices (positive means signal_1 is ahead)
    """

    '''
    # Find peaks in both signals
    peaks_1, _ = find_peaks(signal_1, height=height_1, distance=distance)
    peaks_2, _ = find_peaks(signal_2, height=height_2, distance=distance)

    # Take the first 3
    imu_peaks_3 = peaks_1[:3]
    angle_peaks_3 = peaks_2[:3]

    # Calculate time shift (based on first peak)
    time_shift = imu_peaks_3[0] - angle_peaks_3[0]
    '''
    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x)) if normalize else x
    #print(signal_1.shape)
    
    signal_1 = signal_1[:,:,0]
    signal_1 = signal_1.squeeze(0)
    signal_2 = signal_2[:,:,0]
    signal_2 = signal_2.squeeze(0)

    signal_1 = normalize(signal_1)
    signal_2 = normalize(signal_2)
    #print(signal_1.shape)

    peaks_1, _ = find_peaks(signal_1, prominence=prominence_1, distance=distance)
    peaks_2, _ = find_peaks(signal_2, prominence=prominence_2, distance=distance)

    # Take first 3
    imu_peaks_3 = peaks_1[:3]
    angle_peaks_3 = peaks_2[:3]

    time_shift = imu_peaks_3[0] - angle_peaks_3[0]
    
    
    # Optional plot
    if plot:
        plt.figure(figsize=(14, 5))

        plt.subplot(2, 1, 1)
        plt.plot(signal_1, label='Signal 1 (e.g. IMU)')
        plt.plot(imu_peaks_3, signal_1[imu_peaks_3], 'rx')
        plt.title('Signal 1 - First 3 Peaks')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(signal_2, label='Signal 2 (e.g. Angle)')
        plt.plot(angle_peaks_3, signal_2[angle_peaks_3], 'rx')
        plt.title('Signal 2 - First 3 Peaks')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return imu_peaks_3, angle_peaks_3, time_shift


def butter_low(data, order=4, fc=5, fs=100):
    """
    Zero-lag butterworth filter for column data (i.e. padding occurs along axis 0).
    The defaults are set to be reasonable for standard optoelectronic data.
    """
        
    # Filter design
    b, a = butter(order, 2*fc/fs, 'low')
    # Make sure the padding is neither overkill nor larger than sequence length permits
    padlen = min(int(0.5*data.shape[0]), 200)
    # Zero-phase filtering with symmetric padding at beginning and end
    filt_data = filtfilt(b, a, data, padlen=padlen, axis=1)
    return filt_data


def save_to_csv(path, nn, opt, calib_nn, calib_opt):
    """Creates csv file for results"""
    import pandas as pd

    if path is not None:
        cols = ['RMSE Type', 'Flexion', 'Adduction', 'Rotation']
        df_one = pd.DataFrame(columns=cols)
        df_one.loc[0] = ['NN'] + nn.tolist()
        df_one.loc[1] = ['NN + Opt'] + opt.tolist()
        df_one.loc[2] = ['Calibrated NN'] + calib_nn.tolist()
        df_one.loc[3] = ['Calibrated NN + Opt'] + calib_opt.tolist()
        df_one.to_csv(path+'/RMSE_loss.csv', index=False)

    else:
        pass


def evaluate_result(nn_result, combined_result, gt_angle, result_fldr=None, calib=False):
    """Calculate RMSE result (Neural network, Optimization combined model) by comparing with anatomical markers"""

    def print_result(string, array):
        flex, add, rot = array
        print('%s %.2f (Flexion),  %.2f (Adduction),  %.2f (Rotation)'%(string, flex, add, rot))

    if result_fldr is not None:
        # Save result
        if not osp.exists(result_fldr):
            import os; os.makedirs(result_fldr)
        
        calib_name = "calib_" if calib else ""
        np.save(osp.join(result_fldr, calib_name + "nn_result.npy"), nn_result)
        np.save(osp.join(result_fldr, calib_name + "combined_result.npy"), combined_result)

    if gt_angle is not None:
        gt_angle = gt_angle - gt_angle.mean(axis=1)[:, None, :] if calib else gt_angle
        nn_result = nn_result - nn_result.mean(axis=1)[:, None, :] if calib else nn_result
        combined_result = combined_result - combined_result.mean(axis=1)[:, None, :] if calib else combined_result
        
        rmse_nn_result = np.sqrt(((nn_result - gt_angle)**2).mean(axis=1)).mean(axis=0)
        rmse_opt_result = np.sqrt(((combined_result - gt_angle)**2).mean(axis=1)).mean(axis=0)

    else:
        rmse_nn_result = np.nan
        rmse_opt_result = np.nan
    
    # Print on terminal
    calib_print = '(Calibrated)' if calib else '(Uncalibrated)'
    print_result('Neural Network %s  :'%calib_print, rmse_nn_result)
    print_result('Optimization %s    :'%calib_print, rmse_opt_result)
    return rmse_nn_result, rmse_opt_result


def run_demo(inpt_data, gyro_data, 
            angle_norm_dict, ori_norm_dict, 
            angle_model, ori_model, 
            weight, std_ratio, result_fldr, 
            joint='Knee', leg='Left', gt_angle=None,
             **kwargs):
    
    # if the beginning part of data is not clean, select some specific sequence to estimate
    print(inpt_data.shape)
    print(inpt_gyro.shape)
    print(gt_angle.shape)
    #start, end = 0, -1
    
    if gt_angle.shape[1] == 0:
        raise ValueError("gt_angle has no valid time steps after slicing!")

    start = 0 #3500
    end = gt_angle.shape[1]

    inpt_data = inpt_data[:1, start:end]
    gyro_data = gyro_data[:1, start:end]
    
    if gt_angle is not None:
        gt_angle = gt_angle[:1, start:end]  
    
    # Neural Network Prediction
    with torch.no_grad():
        # normalize input data
        inpt_data_angle = (inpt_data - angle_norm_dict['x_mean']) / angle_norm_dict['x_std']
        inpt_data_ori = (inpt_data - ori_norm_dict['x_mean']) / ori_norm_dict['x_std']
        
        # Predict angle
        angle_model.eval()
        alpha = angle_model(inpt_data_angle)

        # Predict orientation
        ori_model.eval()
        ori_pred = ori_model(inpt_data_ori)
        #print(ori_pred.shape)
        #ori_pred = rot6_to_rotmat(ori_pred)

        # Un-normalize output prediction
        alpha = alpha * angle_norm_dict['y_std'] + angle_norm_dict['y_mean']

        alpha = alpha.detach().cpu().double().numpy()
        ori_pred = ori_pred.detach().cpu().double().numpy()
        
    # Get beta from optimization
    #beta = optimization_demo(ori_pred, gyro_data, joint=joint, leg=leg)

    # Get theta from alpha and beta
    #beta = (beta - beta.mean(axis=1)[:, None]) * std_ratio + alpha.mean(axis=1)[:, None]
    theta = weight * alpha[:, start:end] + (1 - weight) * alpha[:, start:end]

    rmse_nn, rmse_opt = evaluate_result(alpha[:, start:end], theta, gt_angle[:, start:end], 
                                        result_fldr=result_fldr, calib=False)
    print('\n\n')
    rmse_nn_calib, rmse_opt_calib = evaluate_result(alpha[:, start:end], theta, gt_angle[:, start:end], 
                                                result_fldr=result_fldr, calib=True)

    save_to_csv(result_fldr, rmse_nn, rmse_opt, rmse_nn_calib, rmse_opt_calib)
    save_predictions_to_csv_separate(result_fldr, alpha[:, start:end], theta, gt_angle[:, start:end])
        
        

def load_custom_data(path, is_imu_data=True, add_norm=True):
    """Load IMU data from path.
    Assumes numpy array or torch tensor. Adds norm as 4th dimension if add_norm=True.
    
    Parameters:
        path (str): path to .npy or .pkl file
        is_imu_data (bool): whether to treat data as IMU format (B x T x D)
        add_norm (bool): if True, compute and append norm as 4th dimension if D=3

    Returns:
        torch.Tensor or np.ndarray: loaded data
    """
    if path[-3:] == "npy":
        _data = np.load(path, allow_pickle=True).astype(float)
        #length = _data.shape[0]
        #_data = _data[length//2:length]
        _data = torch.from_numpy(_data)
    elif path[-3:] == "pkl":
        with open(path, "rb") as fopen:
            _data = pickle.load(fopen)
            if isinstance(_data, np.ndarray):
                _data = torch.from_numpy(_data)
            else:
                err_msg = "Data type {} is not supported".format(type(_data))
                assert isinstance(_data, torch.Tensor), err_msg
    else:
        err_msg = "Input file format {} is not supported".format(path[-3:])
        raise NotImplementedError(err_msg)

    if len(_data.shape) == 2:
        _data = _data[None]

    if not is_imu_data:
        #length = _data.shape[0]
        #_data = _data[length//2:length]
        return _data.double().numpy() if isinstance(_data, torch.Tensor) else _data

    sz_b, sz_l, sz_d = _data.shape
    assert sz_d in [3, 4], "Dimension of imu data should be 3 or 4"

    if sz_d == 3 and add_norm:
        norm = torch.norm(_data, p='fro', dim=-1, keepdim=True)
        _data = torch.cat([_data, norm], dim=-1)
    return _data #np.array(_data)

def prepare_data(root_path, subject, segment1, segment2, joint, device, dtype):
    
    seg1_accel_path = osp.join(root_path, subject, segment1, "acc.npy")
    seg2_accel_path = osp.join(root_path, subject, segment2, "acc.npy")
    seg1_gyro_path = osp.join(root_path, subject, segment1, "gyr.npy")
    seg2_gyro_path = osp.join(root_path, subject, segment2, "gyr.npy")
    #gt_angle_path = osp.join(root_path, subject, "Left"+str(joint)+"Angle", "angle.npy")
    gt_angle_path = osp.join(root_path, subject, "l"+str(joint).lower(), "angle.npy")

    # Load custom data
    seg1_accel = load_custom_data(seg1_accel_path)
    seg2_accel = load_custom_data(seg2_accel_path)
    seg1_gyro = load_custom_data(seg1_gyro_path)
    seg2_gyro = load_custom_data(seg2_gyro_path)
    
    #seg1_accel = seg1_accel[:,2000:8000,:]
    #seg2_accel = seg2_accel[:,2000:8000,:]
    #seg1_gyro = seg1_gyro[:,2000:8000,:]
    #seg2_gyro = seg2_gyro[:,2000:8000,:]
    
    if gt_angle_path != "":
        gt_angle = load_custom_data(gt_angle_path, is_imu_data=False)
        #gt_angle = gt_angle[:,2000:8000,:]
        
        # Smooth Ground-truth values
        b, a = butter(4, 2*5/100, 'low')
        padlen = min(int(0.5*gt_angle.shape[1]), 200)
        gt_angle = filtfilt(b, a, gt_angle, padlen=padlen, axis=1)
    
    else:
        gt_angle = None
        
    print(seg1_accel.shape)
    print(gt_angle.shape)
    
    #walking_start = 3500
    #walking_end = 21000
        
    #imu_peaks, angle_peaks, offset = find_alignment_peaks(seg1_accel, gt_angle)
    #print(offset)
    #t_angle = gt_angle[:, walking_start:walking_end, :]
    #seg1_accel = torch.from_numpy(seg1_accel[:, walking_start+offset:walking_end+offset, :].astype('float32'))
    #seg2_accel = torch.from_numpy(seg2_accel[:, walking_start+offset:walking_end+offset, :].astype('float32'))
    #seg1_gyro  = torch.from_numpy(seg1_gyro[:, walking_start+offset:walking_end+offset, :].astype('float32'))
    #seg2_gyro  = torch.from_numpy(seg2_gyro[:, walking_start+offset:walking_end+offset, :].astype('float32'))
    
    #print(f"1: {seg1_accel.shape}")
    #print(f"2: {gt_angle.shape}")
    

    inpt_data = torch.cat([seg1_accel, seg1_gyro, seg2_accel, seg2_gyro], dim=-1)
    inpt_data = inpt_data.to(device=device, dtype=dtype)
    
    #print(inpt_data.shape)

    inpt_gyro = torch.cat([seg1_gyro[:, :, :-1], seg2_gyro[:, :, :-1]], dim=-1)
    inpt_gyro = inpt_gyro.double().numpy()
    #print(inpt_gyro.shape)

    return inpt_data, inpt_gyro, gt_angle


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Demo code arguments')
    
    parser.add_argument('--joint', choices=["Knee", "Hip", "Ankle"],
                        type=str, help="The type of joint")

    parser.add_argument('--activity', choices=["Walking", "Running"], 
                        type=str, help="The type of activity")

    parser.add_argument('--root-path', type=str, 
                        help="custom data root path")

    parser.add_argument('--angle-model-fldr', type=str, 
                        default="",
                        help="model folder of angle prediction")
    
    parser.add_argument('--ori-model-fldr', type=str, 
                        default="",
                        help="model folder of orientation prediction")

    parser.add_argument('--result-fldr', type=str, 
                        default="/Users/ccy/Documents/CMU/Spring2025/42696 Wearable Health Technology/Projects/wearable-project/data preprocessing/result",
                        help="folder to save result files")    

    parser.add_argument('--use-cuda', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='cuda configuration')

    args = parser.parse_args()

    dtype = torch.float
    device = 'cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu'
    
    result_fldr = args.result_fldr
    joint = args.joint
    activity = args.activity
    root_path = os.path.join(args.root_path, activity)
    subject = '06'    # Select the direction of your target leg
    segment1 = 'lthigh'
    segment2 = 'lshank'
    
    inpt_data, inpt_gyro, gt_angle = prepare_data(root_path, subject, segment1, segment2, joint, device, dtype)
    
    angle_model_fldr = osp.join(args.angle_model_fldr, activity, joint)
    ori_model_fldr = osp.join(args.ori_model_fldr, activity, joint)

    # Load prediction model
    for model_fldr in [angle_model_fldr , ori_model_fldr]:
        #print(model_fldr)
        #_, model, _ = next(os.walk(model_fldr))
        #print(model[0])
        #model_fldr_ = osp.join(model_fldr, model[0])
        with open(osp.join(angle_model_fldr, "model_kwargs.pkl"), "rb") as fopen:
            model_kwargs = pickle.load(fopen)
            print(model_kwargs)
        model = globals()['CustomConv1D'](**model_kwargs) if model_kwargs["model_type"] == "CustomConv1D" \
                                                        else globals()['transformer'](**model_kwargs)
        state_dict = torch.load(osp.join(angle_model_fldr, "model.pt"), map_location=torch.device('cpu'),weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device=device, dtype=dtype)

        if model_fldr == angle_model_fldr:
            angle_model = model
            angle_norm_dict = torch.load(osp.join(angle_model_fldr, "norm_dict.pt"), map_location=torch.device('cpu'),weights_only=True)['params']

        else:
            ori_model = model
            ori_norm_dict = torch.load(osp.join(ori_model_fldr, "norm_dict.pt"), map_location=torch.device('cpu'), weights_only=True)['params']        

    # Get optimization parameters (weight, std ratio)
    #with open('Data/5_Optimization/parameters.pkl', 'rb') as fopen:
        #params = pickle.load(fopen)
    std_ratio = 1 #params['%s_%s_std'%(joint, activity)]
    weight = 1 #params['%s_%s_weight'%(joint, activity)]

    run_demo(inpt_data, inpt_gyro, angle_norm_dict, 
            ori_norm_dict, angle_model, 
            ori_model, weight, std_ratio, result_fldr, 
            joint=joint, leg=segment1, gt_angle=gt_angle)
    
    