# -------------------------
#
# Functions for use in hyperopt parameter optimization including
# establishing search space, assigning model params, and training/scoring functions
#
# -------------------------

from utils.eval_utils import load_predictions, calc_rmses
import sys; sys.path.append('./')
from nn_models import train_model
from copy import copy
import numpy as np
from collections import defaultdict
import pandas as pd
from functools import partial
import torch
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

def search_space(model):
    # Creates a dictionary of the parameters and the spaces to search over

    hidden_layers = [10, 20, 30, 40, 50, 60, 70]

    # Creates parameter space for general parameters
    space = {
        'lr': hp.uniform('lr', 0.00001, 0.01),           # Uniform learning rate
        'lr_decay': hp.uniform('lr_decay', 0.9000, 0.9997), # Uniform learning rate decay
        'iter': hp.choice('iter', [500, 1000, 5000]),      # Batch iteration count
        'dropout': hp.uniform('dropout', 0.0, 0.3),         # Uniform dropout value
        'num_layers': hp.choice('num_layers', [1, 2, 3]),    # Number of hidden layers
        'size_layers': hp.choice('size_layers', hidden_layers) # Size of hidden layers
    }

    # Creates parameter space for specific models
    if model == 'CustomConv1D':
        space['window'] = hp.choice('window', [11, 21, 31, 41, 51])  # Window size
    elif model == 'CustomLSTM':
        space['bidir'] = hp.choice('bidir', [False, True])  # Directionality
    # elif model == 'transformer':
    #     # Example search space for the Transformer
    #     # You can enlarge or reduce this space as needed:
    #     space['d_model'] = hp.choice('d_model', [64, 96, 128])
    #     space['num_heads'] = hp.choice('num_heads', [2, 4, 6])
    #     space['dim_feedforward'] = hp.choice('dim_feedforward', [128, 256, 512])
    #     space['num_layers'] = hp.choice('num_layers', [2, 4, 6])
    #     space['dropout'] = hp.uniform('dropout', 0.0, 0.3)
    #     # If you want a "window" hyperparam, you could add that too
    #     # but typically for Transformers you might rely on seq_len from train spec
    # return space
    # elif model == 'transformer':
    #     space['d_model'] = hp.choice('d_model', [64, 96, 128])          # smaller embedding sizes
    #     space['num_heads'] = hp.choice('num_heads', [2, 4])            # fewer heads (avoid 6+ if OOM)
    #     space['dim_feedforward'] = hp.choice('dim_feedforward', [128, 256, 512])  # keep feedforward small
    #     space['num_layers'] = hp.choice('num_layers', [2, 3, 4])       # keep depth modest
    #     space['dropout'] = hp.uniform('dropout', 0.0, 0.3)
    # return space    
    elif model == 'transformer':
        # Keep everything minimal:
        space['d_model'] = hp.choice('d_model', [32, 64])     # smaller embeddings
        space['num_heads'] = hp.choice('num_heads', [1, 2])   # just 1–2 heads
        space['dim_feedforward'] = hp.choice('dim_feedforward', [64, 128])  # smaller feedforward
        space['num_layers'] = hp.choice('num_layers', [1, 2]) # fewer layers
        space['dropout'] = hp.uniform('dropout', 0.0, 0.3)
    return space    

def assign_params(params, model, joint, data_path, result_path):
    # Assigns the parameters to the general and model specifications

    # General specification
    general_spec = {
        'model_type': model,
        'data_path': data_path, 
        'result_path': result_path + joint[0].upper() + joint[1:] + '/',
        # Model input and output specification
        'inp': ['pelvis', 'thigh', 'shank', 'foot'], 
        'outp': ['hip', 'knee', 'ankle'],
        # Logging specification
        'log_metrics': ['loss', 'lr', 'rmse'], 
        'log_freq': 100, 
        'check_freq': 1000,
        # Optimizer and learning rate schedule
        'optim': 'Adam', 
        'lr': params['lr'], 
        'loss': 'MSELoss', 
        'scheduler': 'ExponentialLR',
        'lr_decay': params['lr_decay'],  
        'prog': True,
        # Data augmentation
        'aug': True, 
        'rot_type': 'normal', 
        'rot_spread': 0.075,
        'x_noise': 0.15, 
        'y_noise': 0,
        # Training schedule
        'seq_len':  [50, 100], #[200, 400], 
        'num_iter': 2 * [params['iter']],
        'batch_size': [2, 4] # 8, 16
    }

    # Assigns inputs according to the joint analyzed
    if joint == 'hip':
        general_spec['inp'] = ['pelvis', 'thigh']
    elif joint == 'knee':
        general_spec['inp'] = ['thigh', 'shank']
    elif joint == 'ankle':
        general_spec['inp'] = ['shank', 'foot']

    general_spec['outp'] = [joint]

    # Model parameters and assigns name string
    if model == 'CustomLSTM':
        model_spec = {
            'inp_size': [8 * len(general_spec['inp'])],
            'outp_size': [3 * len(general_spec['outp'])],
            'layers': params['num_layers'] * [params['size_layers']],
            'dropout': params['num_layers'] * [params['dropout']],
            'bidir': params['bidir']
        }
        tmp_name = 'lstm_bidir' if params['bidir'] else 'lstm_unidir'
        mdl_name = '{}_layers{}_size{}_drop{}_lr{}_decay{}_iter{}'.format(
            tmp_name, params['num_layers'], params['size_layers'],
            params['dropout'], params['lr'],
            params['lr_decay'], params['iter']
        )
    elif model == 'CustomConv1D':
        model_spec = {
            'inp_size': [8 * len(general_spec['inp'])],
            'outp_size': [3 * len(general_spec['outp'])],
            'layers': params['num_layers'] * [params['size_layers']],
            'window': params['window'], 
            'groups': params['num_layers'] * [1],
            'conv_activation': params['num_layers'] * ['ReLU'],
            'conv_dropout': params['num_layers'] * [params['dropout']],
            'conv_batchnorm': True,
            'lin_layers': [30],
            'lin_activation': ['Sigmoid'],
            'lin_dropout': []
        }
        model_spec['lin_layers'] = [max(20, 10 * len(general_spec['outp']))]
        tmp_name = 'conv1d'
        mdl_name = '{}_layers{}_size{}_win{}_drop{}_lr{}_decay{}_iter{}'.format(
            tmp_name, params['num_layers'], params['size_layers'],
            params['window'], params['dropout'], params['lr'],
            params['lr_decay'], params['iter']
        )
        model_spec['inp_size'] = [8 * len(general_spec['inp'])]
        model_spec['outp_size'] = [3 * len(general_spec['outp'])]
    elif model == 'transformer':
        # model_spec = {
        #     'inp_size': [8 * len(general_spec['inp'])],
        #     'outp_size': [3 * len(general_spec['outp'])],
        #     'num_layers': params['num_layers'],
        #     'd_model': params['d_model'],
        #     'num_heads': params['num_heads'],
        #     'dim_feedforward': params['dim_feedforward'],
        #     'dropout': params['dropout'],
        #     'max_seq_len': 400,  # or 200—depending on largest seq in your schedule
        #     # Make sure you pass 'prediction' to keep the same pipeline
        #     'prediction': 'angle'
        # }


        # model_spec = {
        #     'inp_size': [8 * len(general_spec['inp'])],
        #     'outp_size': [3 * len(general_spec['outp'])],
        #     'num_layers': params['num_layers'],        # e.g., in [2, 3, 4]
        #     'd_model': params['d_model'],             # e.g., in [64, 96, 128]
        #     'num_heads': params['num_heads'],         # e.g., in [2, 4]
        #     'dim_feedforward': params['dim_feedforward'],  # e.g., in [128, 256, 512]
        #     'dropout': params['dropout'],             # e.g., 0.0 ~ 0.3
        #     'max_seq_len': 200,                       # or even 100 if 400 is too large
        #     'prediction': 'angle'
        # }

        model_spec = {
            'inp_size': [8 * len(general_spec['inp'])],
            'outp_size': [3 * len(general_spec['outp'])],
            'num_layers': params['num_layers'],
            'd_model': params['d_model'],
            'num_heads': params['num_heads'],
            'dim_feedforward': params['dim_feedforward'],
            'dropout': params['dropout'],
            # Try a smaller max_seq_len (e.g., 80 or 100):
            'max_seq_len': 100,
            'prediction': 'angle'
        }

        mdl_name = ("transformer_layers{}_dmodel{}_heads{}_ff{}_drop{}_lr{}_decay{}_iter{}"
                    .format(params['num_layers'], params['d_model'], params['num_heads'],
                            params['dim_feedforward'], params['dropout'], params['lr'],
                            params['lr_decay'], params['iter']))
    # end if
    general_spec['name'] = mdl_name
    model_spec['prediction'] = 'angle'

    return general_spec, model_spec

def obj_fcn(params, model, joint, data_path, result_path):
    # Returns score from training model - averaging RMSE values for all degrees of freedom

    # Assigns parameters to variables
    general_spec, model_spec = assign_params(params, model, joint, data_path, result_path)

    # Train model with current parameters
    from copy import copy
    train_model.main(copy(general_spec), copy(model_spec), show_bar=False)
    torch.cuda.empty_cache()

    # Calculate average RMSE from validation data by loading predictions
    model_path = general_spec['result_path'] + general_spec['name']
    (x_val, y_val, y_pred_val, x_test, y_test, y_pred_test) = load_predictions(model_path)
    val_rmse = np.mean(np.mean(calc_rmses(y_val, y_pred_val), axis=0))
    
    return {'loss': val_rmse, 'status': STATUS_OK, 'params': params}

def run_model(data_path, result_path, num_eval, model, joint_list, rstate=None):
    # Runs the hyperparameter optimization

    # Set default random seed if no custom random state is provided
    seed = 100
    if rstate is None:
        rstate = np.random.RandomState(seed)

    # Create trials objects for each joint
    trials_hip = Trials()
    trials_knee = Trials()
    trials_ankle = Trials()
    best_models = []

    # Optimize parameters for each joint
    for joint in joint_list:
        if joint == 'hip':
            trials = trials_hip
        elif joint == 'knee':
            trials = trials_knee
        elif joint == 'ankle':
            trials = trials_ankle

        # Create search space
        space = search_space(model)

        # Perform hyperparameter optimization
        best_model = fmin(
            fn=partial(obj_fcn, model=model, joint=joint,
                       data_path=data_path, result_path=result_path),
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=num_eval,
            rstate=rstate,
            show_progressbar=True
        )

        print('Minimum RMSE score attained for {}: {:.4f}'.format(
            joint, trials.best_trial['result']['loss']))

        # Create dataframe of all parameters for each trial
        params_list = [p['params'] for p in trials.results]
        from collections import defaultdict
        param_dict = defaultdict(list)
        for k, v in ((k, v) for d in params_list for k, v in d.items()):
            param_dict[k].append(v)
        param_df = pd.DataFrame.from_dict(param_dict)
        loss_dict = {'loss': [x['loss'] for x in trials.results]}
        loss_df = pd.DataFrame.from_dict(loss_dict)
        result_df = pd.concat([loss_df, param_df], axis=1)

        # Export loss data to CSV file
        file_path = result_path + joint + '_loss_results_' + model + '_.csv'
        result_df.to_csv(file_path, index=True)
        print('File saved: ' + file_path)
        print()

        best_models.append(best_model)

    return best_models

if __name__ == '__main__':
    pass
