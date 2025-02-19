import h5py
import numpy as np
import os

# 設定 HDF5 檔案路徑
h5_path = "C:/Users/User/JointAnglePrediction_JOB/Data/2_Processed/walking_data.h5"  # 也可修改為 running_data.h5
output_dir = "C:/Users/User/JointAnglePrediction_JOB/extracted_numpy"

# 🔹 指定要處理的 subjects（填入你要的 subject ID）
selected_subjects = ["s700"]  # 🛠️ 修改這裡來選擇特定的 subjects

# 確保輸出資料夾存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 讀取 HDF5 檔案
with h5py.File(h5_path, "r") as f:
    for subject in selected_subjects:
        if subject not in f:
            print(f"⚠️ 警告: {subject} 不存在於 .h5 檔案內，跳過處理。")
            continue

        print(f"📂 讀取 Subject: {subject}")
        subject_dir = os.path.join(output_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)

        for sensor in f[subject].keys():  # ex: lhip, rfoot, pelvis
            sensor_dir = os.path.join(subject_dir, sensor)
            os.makedirs(sensor_dir, exist_ok=True)
            
            for data_type in f[subject][sensor].keys():  # ex: acc, gyr, angle
                dataset_path = f"{subject}/{sensor}/{data_type}"
                print(f"   📄 讀取 {dataset_path}")

                # 讀取數據
                data = f[dataset_path][:]
                
                # 存成 .npy 檔案
                npy_filename = os.path.join(sensor_dir, f"{data_type}.npy")
                np.save(npy_filename, data)
                print(f"   ✅ 已儲存: {npy_filename}")

print("\n🚀 指定的 subjects 資料已成功轉存為 .npy，請檢查資料夾！")
