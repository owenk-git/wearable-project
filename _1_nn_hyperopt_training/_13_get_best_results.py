import xlsxwriter
import numpy as np
import os
import shutil
from utils import eval_utils
from glob import glob
import pandas as pd

if __name__ == '__main__':

    base_path = '/home/owen/projects/25spring/wearable/Data/3_Hyperopt_Results'
    new_path = '/home/owen/projects/25spring/wearable/Data/4_Best_Hyperopt'

    act_list = ['Walking', 'Running']
    sub_folders = ['Hip', 'Knee', 'Ankle']

    for tfldr in act_list:
        task_path = os.path.join(base_path, tfldr)
        for sfldr in sub_folders:
            # Initialize RMSE list and valid model paths list
            rmse_list = []
            valid_model_paths = []
            result_path = os.path.join(task_path, sfldr)
            
            # Create destination folder if it doesn't exist
            dest_folder = os.path.join(new_path, tfldr, sfldr)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            
            # Get all model paths in this folder
            model_paths = glob(os.path.join(result_path, '*'))
            model_paths.sort()
            
            # Evaluate each model's RMSE on the validation set
            for model in model_paths:
                try:
                    (x_val, y_val, y_pred_val,
                     x_test, y_test, y_pred_test) = eval_utils.load_predictions(model)
                    val_rmse = eval_utils.calc_rmses(y_val, y_pred_val)
                    rmse_list.append(np.mean(val_rmse, axis=0, keepdims=True))
                    valid_model_paths.append(model)
                except FileNotFoundError:
                    print(f"Skipping model {model} due to missing predictions files.")
                    continue

            if not valid_model_paths:
                print(f"No valid models found for {tfldr}/{sfldr}. Skipping folder.")
                continue

            # Determine the best model based on the validation score
            rmses = np.concatenate(rmse_list, axis=0)
            score = np.mean(rmses, axis=1, keepdims=True)
            best_idx = np.argmin(score)
            best_model_path = valid_model_paths[best_idx]
            best_model_name = os.path.basename(best_model_path)
            dest = os.path.join(dest_folder, best_model_name)

            if os.path.exists(dest):
                print(f"Destination {dest} already exists. Skipping copying of best model.")
            else:
                shutil.copytree(best_model_path, dest)

    #
    # Create spreadsheet summary
    #

    col_angles = ['Score', 'Flex', 'Add', 'Rot']
    angle_metrics = ['Mean', 'Median', 'Std']
    model_strings = ['conv*', 'lstm*']

    for act in act_list:
        result_path = os.path.join(new_path, act)
        writer = pd.ExcelWriter(os.path.join(result_path, '_model_summaries.xlsx'), engine='xlsxwriter')

        for fldr in sub_folders:
            curr_num_rows = 0
            sheet_name = fldr

            for model_str in model_strings:
                # Create empty dataframes for validation and test summaries
                val_df = eval_utils.create_hyperopt_df([fldr], col_angles, angle_metrics)
                test_df = eval_utils.create_hyperopt_df([fldr], col_angles, angle_metrics)

                pth = os.path.join(result_path, fldr)
                model_paths = glob(os.path.join(pth, model_str))
                model_paths.sort()
                for model in model_paths:
                    try:
                        model_name = os.path.basename(model)
                        x_val, y_val, y_pred_val, x_test, y_test, y_pred_test = \
                            eval_utils.load_predictions(model)
                        val_rmse = eval_utils.calc_rmses(y_val, y_pred_val)
                        test_rmse = eval_utils.calc_rmses(y_test, y_pred_test)

                        # Update the dataframes with RMSE summaries
                        val_df = eval_utils.add_hyperopt_summary(
                            val_df, val_rmse, model_name, [fldr], col_angles, angle_metrics)
                        test_df = eval_utils.add_hyperopt_summary(
                            test_df, test_rmse, model_name, [fldr], col_angles, angle_metrics)
                    except FileNotFoundError:
                        print(f"Skipping model {model} in summary due to missing predictions files.")
                        continue

                # Write validation and test summaries side by side
                val_df.to_excel(writer, sheet_name=sheet_name, startrow=curr_num_rows, startcol=0)
                test_df.to_excel(writer, sheet_name=sheet_name, startrow=curr_num_rows,
                                 startcol=val_df.shape[1] + 5)
                curr_num_rows += val_df.shape[0] + 9  # Adjust spacing between sections as needed

        # writer.save()
        writer.close()
        print('Best models extracted for {}!'.format(act))


# # -------------------------
# #
# # Retrieves best model from all trained hyperopt models
# #
# # --------------------------

# import numpy as np
# import os
# import shutil
# from utils import eval_utils
# from glob import glob
# import pandas as pd


# if __name__ == '__main__':

#     base_path = '/home/owen/projects/25spring/wearable/Data/3_Hyperopt_Results'
#     new_path = '/home/owen/projects/25spring/wearable/Data/4_Best_Hyperopt'

#     # res_path = '/home/owen/projects/25spring/wearable/Data/3_Hyperopt_Results/'
#     # walking_result_path = '/home/owen/projects/25spring/wearable/Data/3_Hyperopt_Results/Walking/'
#     # running_result_path = '/home/owen/projects/25spring/wearable/Data/3_Hyperopt_Results/Running/'

#     act_list = ['Walking', 'Running']
#     sub_folders = ['Hip', 'Knee', 'Ankle']

#     for tfldr in act_list:
#         task_path = os.path.join(base_path, tfldr)
#         for sfldr in sub_folders:
#             # Initialize RMSE list
#             rmse_list = []
#             # Get current path
#             result_path = os.path.join(task_path, sfldr)
#             # Create folder for best model if it doesn't exist already
#             if not os.path.exists(os.path.join(new_path, tfldr, sfldr)):
#                 os.makedirs(os.path.join(new_path, tfldr, sfldr))
#             # Get all model paths for this folder
#             model_paths = glob(os.path.join(result_path, '*'))
#             model_paths.sort()
#             # Get RMSEs for all models
#             for model in model_paths:
#                 (x_val, y_val, y_pred_val,
#                  x_test, y_test, y_pred_test) = eval_utils.load_predictions(model)
#                 val_rmse = eval_utils.calc_rmses(y_val, y_pred_val)
#                 rmse_list.append(np.mean(val_rmse, axis=0, keepdims=True))
#             # Create RMSE array
#             rmses = np.concatenate(rmse_list, axis=0)
#             score = np.mean(rmses, axis=1, keepdims=True)
#             # Get index of best models
#             best_idx = np.argmin(score)
#             best_model_name = model_paths[best_idx].split('/')[-1]
#             dest = os.path.join(new_path, tfldr, sfldr, best_model_name)
#             shutil.copytree(model_paths[best_idx], dest)

#     #
#     # Creates spreadsheet summary
#     #

#     col_angles = ['Score', 'Flex', 'Add', 'Rot']
#     angle_metrics = ['Mean', 'Median', 'Std']
#     model_strings = ['conv*', 'lstm*']

#     for act in act_list:
#         result_path = new_path+'/'+act+'/'

#         writer = pd.ExcelWriter(result_path + '_model_summaries.xlsx', engine='xlsxwriter')

#         for fldr in sub_folders:
#             curr_num_rows = 0
#             for i, model_str in enumerate(model_strings):
#                 # Creates dataframe of validation and test data
#                 val_df = eval_utils.create_hyperopt_df(
#                     [fldr], col_angles, angle_metrics)
#                 test_df = eval_utils.create_hyperopt_df(
#                     [fldr], col_angles, angle_metrics)

#                 pth = result_path + fldr + '/'
#                 # Summarize all models
#                 model_paths = glob(pth + model_str)
#                 model_paths.sort()
#                 for model in model_paths:
#                     model_name = model.split('/')[-1]

#                     x_val, y_val, y_pred_val, x_test, y_test, y_pred_test = \
#                         eval_utils.load_predictions(model)
#                     val_rmse = eval_utils.calc_rmses(y_val, y_pred_val)
#                     test_rmse = eval_utils.calc_rmses(y_test, y_pred_test)

#                     # Adds validation RMSE data to spreadsheet
#                     val_df = eval_utils.add_hyperopt_summary(
#                         val_df, val_rmse, model_name, [fldr], col_angles, angle_metrics)
#                     # Adds test RMSE data to spreadsheet
#                     test_df = eval_utils.add_hyperopt_summary(
#                         test_df, test_rmse, model_name, [fldr], col_angles, angle_metrics)

#                 # Write to excel (validation on left, test on right)
#                 val_df.to_excel(writer, sheet_name=fldr, startrow=curr_num_rows, startcol=0)
#                 test_df.to_excel(writer, sheet_name=fldr, startrow=curr_num_rows,
#                                  startcol=val_df.shape[1] + 5)
#                 # Update start index for the rows
#                 curr_num_rows += val_df.shape[0] + 4 + 5

#         writer.save()

#         print('Best models extracted for {}!'.format(act))