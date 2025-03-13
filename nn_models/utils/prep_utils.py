# -------------------------
#
# Functions to prepare for model training
#
# -------------------------

import numpy as np
import torch
import h5py
from math import ceil
from numpy.random import RandomState

def get_sub_dict(h5path, subject_ids, inp_fields, outp_fields, prediction, device):
    # Create dictionary of subject data

    if not isinstance(inp_fields, list):
        inp_fields = [inp_fields]
    if not isinstance(outp_fields, list):
        outp_fields = [outp_fields]

    sub_dict = {}
    with h5py.File(h5path, 'r') as fh:
        for eid, sid in enumerate(subject_ids):
            # Gather inputs from subject data
            y_ori_list, x_list = [], []
            for inp in inp_fields:
                # Checks if segments are labelled as left/right
                if inp[0] != 'r' and inp[0] != 'l' and not (inp == 'pelvis'):
                    # ---------------------------
                    # Process bilateral (non-pelvis) segments:
                    # ---------------------------
                    # Right accelerations
                    right_acc = fh['s' + str(sid) + '/r' + inp + '/acc'][:, :]
                    tmp = np.linalg.norm(right_acc, axis=1, keepdims=True)
                    right_acc = np.concatenate((right_acc, tmp), axis=1)
                    # Right angular velocities
                    right_gyr = fh['s' + str(sid) + '/r' + inp + '/gyr'][:, :]
                    tmp = np.linalg.norm(right_gyr, axis=1, keepdims=True)
                    right_gyr = np.concatenate((right_gyr, tmp), axis=1)
                    # Left accelerations
                    left_acc = fh['s' + str(sid) + '/l' + inp + '/acc'][:, :]
                    tmp = np.linalg.norm(left_acc, axis=1, keepdims=True)
                    left_acc = np.concatenate((left_acc, tmp), axis=1)
                    # Left angular velocities
                    left_gyr = fh['s' + str(sid) + '/l' + inp + '/gyr'][:, :]
                    tmp = np.linalg.norm(left_gyr, axis=1, keepdims=True)
                    left_gyr = np.concatenate((left_gyr, tmp), axis=1)
                    
                    # Right orientations
                    key_right_ori = 's' + str(sid) + '/r' + inp + '/rmat'
                    if key_right_ori in fh:
                        right_ori = fh[key_right_ori][:, :]
                    else:
                        print("Warning: key {} not found for subject {}".format(key_right_ori, sid))
                        right_ori = None
                    # Left orientations
                    key_left_ori = 's' + str(sid) + '/l' + inp + '/rmat'
                    if key_left_ori in fh:
                        left_ori = fh[key_left_ori][:, :]
                    else:
                        print("Warning: key {} not found for subject {}".format(key_left_ori, sid))
                        left_ori = None

                    # Stack right and left data for sensors
                    right_tmp = np.concatenate((right_acc, right_gyr), axis=1)
                    left_tmp = np.concatenate((left_acc, left_gyr), axis=1)

                    x_list.append(np.stack((right_tmp, left_tmp), axis=0))
                    
                    # Only add orientation data if both right and left orientations are available
                    if right_ori is not None and left_ori is not None:
                        y_ori_list.append(np.stack((right_ori, left_ori), axis=0))
                    else:
                        print("Skipping orientation for subject {} input {} due to missing data.".format(sid, inp))
                else:
                    # ---------------------------
                    # Process unilateral segments (e.g., pelvis or segments already labelled as left/right)
                    # ---------------------------
                    # Accelerations
                    acc = fh['s' + str(sid) + '/' + inp + '/acc'][:, :]
                    tmp = np.linalg.norm(acc, axis=1, keepdims=True)
                    acc = np.concatenate((acc, tmp), axis=1)
                    # Angular velocities
                    gyr = fh['s' + str(sid) + '/' + inp + '/gyr'][:, :]
                    tmp = np.linalg.norm(gyr, axis=1, keepdims=True)
                    gyr = np.concatenate((gyr, tmp), axis=1)
                    
                    # Orientations
                    key_ori = 's' + str(sid) + '/' + inp + '/rmat'
                    if key_ori in fh:
                        ori = fh[key_ori][:, :]
                    else:
                        print("Warning: key {} not found for subject {}".format(key_ori, sid))
                        ori = None

                    tmp_concat = np.concatenate((acc, gyr), axis=1)
                    if inp == 'pelvis' and ('thigh' in inp_fields or 'shank' in inp_fields or 'foot' in inp_fields):
                        tmp_stack = np.stack((tmp_concat, tmp_concat), axis=0)
                        if ori is not None:
                            ori_stack = np.stack((ori, ori), axis=0)
                        else:
                            print("Skipping orientation for subject {} input {} due to missing data.".format(sid, inp))
                            ori_stack = None
                        x_list.append(tmp_stack)
                        if ori_stack is not None:
                            y_ori_list.append(ori_stack)
                    else:
                        x_list.append(tmp_concat[None, :, :])
                        if ori is not None:
                            y_ori_list.append(ori[None, :, :])
                        else:
                            print("Skipping orientation for subject {} input {} due to missing data.".format(sid, inp))
            # Concatenate inputs along sensor dimension
            x = np.concatenate(x_list, axis=2)
            
            # Process output (angles)
            y_angle_list = []
            for outp in outp_fields:
                if outp[0] != 'r' and outp[0] != 'l':
                    right_angle = fh['s' + str(sid) + '/r' + outp + '/angle'][:, :]
                    left_angle = fh['s' + str(sid) + '/l' + outp + '/angle'][:, :]
                    tmp = np.stack((right_angle, left_angle), axis=0)
                    y_angle_list.append(tmp)
                else:
                    angle = fh['s' + str(sid) + '/' + outp + '/angle'][:, :]
                    y_angle_list.append(angle[None, :, :])
            
            y_ori = None
            if len(y_ori_list) > 0:
                try:
                    y_ori = np.concatenate(y_ori_list, axis=-1)
                except Exception as e:
                    print("Error concatenating orientation data for subject {}: {}".format(sid, e))
            y_angle = np.concatenate(y_angle_list, axis=2)
            
            if prediction == 'angle':
                y = y_angle.copy()
            elif prediction == 'orientation' and y_ori is not None:
                y = y_ori.copy()
            else:
                print("No orientation data available for subject {}".format(sid))
                y = y_angle.copy()
            
            # Convert to torch tensors and assign to device
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            sub_dict[eid] = [x, y]
            
    return sub_dict


def get_subject_split(h5path, split):
    # Splits dataset into train, validation, and test datasets

    sub_list = []
    with h5py.File(h5path, 'r') as fh:
        subs = list(fh.keys())
        subs.sort(key=lambda x: int(x[1:]))
        for sub in subs:
            if fh[sub].attrs['checks_passed']:
                sub_list.append(sub)
    split_rnd_gen = RandomState(42)
    ids = [int(x[1:]) for x in sub_list]
    split_rnd_gen.shuffle(ids)
    split_ids = [ceil(split[0] * len(ids)), ceil((split[0] + split[1]) * len(ids))]
    train_ids = ids[0:split_ids[0]]
    val_ids = ids[split_ids[0]:split_ids[1]]
    test_ids = ids[split_ids[1]:]

    return train_ids, val_ids, test_ids


def get_device(show_bar=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if show_bar:
        print('Using device:', device)
        print()
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024**3, 1), 'GB')
    return device


def get_cpu():
    device = torch.device('cpu')
    return device


def parse_scheduler(general_spec):
    scheduler_spec = {}
    scheduler_type = general_spec['scheduler']
    scheduler_spec['type'] = scheduler_type
    if scheduler_type == 'ExponentialLR':
        arg_dict = {'gamma': general_spec['lr_decay']}
        scheduler_spec['args'] = arg_dict
    return scheduler_spec


def parse_stages(general_spec):
    seq_lens = general_spec['seq_len']
    batch_sizes = general_spec['batch_size']
    num_iters = general_spec['num_iter']
    assert (len(seq_lens) == len(batch_sizes) and len(seq_lens) == len(num_iters)), 'Numbers of stages is inconsistent'
    num_stages = len(seq_lens)
    stage_spec = {}
    for i in range(num_stages):
        stage_spec[i] = {
            'seq_len': seq_lens[i],
            'batch_size': batch_sizes[i],
            'num_iter': num_iters[i],
            'disprog': not general_spec['prog']
        }
    return stage_spec
