import numpy as np

# Example: Using a custom RandomState subclass (with an added integers method, if needed)
class CustomRandomState(np.random.RandomState):
    def integers(self, low, high=None, size=None, dtype=np.int64):
        return self.randint(low, high, size=size, dtype=dtype)

custom_rstate = CustomRandomState()

from utils.hyperopt_utils import run_model
if __name__ == '__main__':
    # walking_data_path = 'Data/2_Processed/walking_data.h5'
    # walking_result_path = 'Data/3_Hyperopt_Results/Walking/'
    # running_data_path = 'Data/2_Processed/running_data.h5'
    # running_result_path = 'Data/3_Hyperopt_Results/Running/'

    walking_data_path = '/mnt/storage/owen/JointAnglePrediction_JOB/Data/2_Processed/walking_data.h5'
    walking_result_path = '/mnt/storage/owen/JointAnglePrediction_JOB/Data/3_Hyperopt_Results/Walking/'
    running_data_path = '/mnt/storage/owen/JointAnglePrediction_JOB/Data/2_Processed/running_data.h5'
    running_result_path = '/mnt/storage/owen/JointAnglePrediction_JOB/Data/3_Hyperopt_Results/Running/'


    # conv_model = 'CustomConv1D'
    # num_eval_conv = 300


    # print('Walking convolution begin...')
    # walking_conv_hip = run_model(walking_data_path, walking_result_path, num_eval_conv, conv_model, ['hip'], rstate=custom_rstate)
    # walking_conv_knee = run_model(walking_data_path, walking_result_path, num_eval_conv, conv_model, ['knee'], rstate=custom_rstate)
    # walking_conv_ankle = run_model(walking_data_path, walking_result_path, num_eval_conv, conv_model, ['ankle'], rstate=custom_rstate)
    # print('Walking convolution complete!\n')

    # print('Running convolution begin...')
    # running_conv_hip = run_model(running_data_path, running_result_path, num_eval_conv, conv_model, ['hip'], rstate=custom_rstate)
    # running_conv_knee = run_model(running_data_path, running_result_path, num_eval_conv, conv_model, ['knee'], rstate=custom_rstate)
    # running_conv_ankle = run_model(running_data_path, running_result_path, num_eval_conv, conv_model, ['ankle'], rstate=custom_rstate)
    # print('Running convolution complete!\n')

    # # Now for LSTM models
    # lstm_model = 'CustomLSTM'
    # num_eval_lstm = 200

    # print('Walking LSTM begin...')
    # walking_lstm_hip = run_model(walking_data_path, walking_result_path, num_eval_lstm, lstm_model, ['hip'], rstate=custom_rstate)
    # walking_lstm_knee = run_model(walking_data_path, walking_result_path, num_eval_lstm, lstm_model, ['knee'], rstate=custom_rstate)
    # walking_lstm_ankle = run_model(walking_data_path, walking_result_path, num_eval_lstm, lstm_model, ['ankle'], rstate=custom_rstate)
    # print('Walking LSTM complete!\n')

    # print('Running LSTM begin...')
    # running_lstm_hip = run_model(running_data_path, running_result_path, num_eval_lstm, lstm_model, ['hip'], rstate=custom_rstate)
    # running_lstm_knee = run_model(running_data_path, running_result_path, num_eval_lstm, lstm_model, ['knee'], rstate=custom_rstate)
    # running_lstm_ankle = run_model(running_data_path, running_result_path, num_eval_lstm, lstm_model, ['ankle'], rstate=custom_rstate)
    # print('Running LSTM complete!\n')


    # Example usage for the new transformer:
    transformer_model = 'transformer'
    num_eval_transformer = 10

    print('Walking transformer begin...')
    walking_lstm_hip = run_model(walking_data_path, walking_result_path, num_eval_transformer, transformer_model, ['hip'], rstate=custom_rstate)
    walking_lstm_knee = run_model(walking_data_path, walking_result_path, num_eval_transformer, transformer_model, ['knee'], rstate=custom_rstate)
    walking_lstm_ankle = run_model(walking_data_path, walking_result_path, num_eval_transformer, transformer_model, ['ankle'], rstate=custom_rstate)
    print('Walking transformer complete!\n')

    print('Running transformer begin...')
    running_lstm_hip = run_model(running_data_path, running_result_path, num_eval_transformer, transformer_model, ['hip'], rstate=custom_rstate)
    running_lstm_knee = run_model(running_data_path, running_result_path, num_eval_transformer, transformer_model, ['knee'], rstate=custom_rstate)
    running_lstm_ankle = run_model(running_data_path, running_result_path, num_eval_transformer, transformer_model, ['ankle'], rstate=custom_rstate)
    print('Running transformer complete!\n')
