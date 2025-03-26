# -*- coding: utf-8 -*-
"""
Single-subject (SUB01) IMU data merger.

Folders inside SUB01:
    LeftAnkleAngle/angle.npy    (frames, 3)
    LeftFoot/acc.npy            (frames, 3)
    LeftFoot/gyr.npy            (frames, 3)
    LeftHipAngle/angle.npy      (frames, 3)
    LeftShank/acc.npy           (frames, 3)
    LeftShank/gyr.npy           (frames, 3)
    LeftThigh/acc.npy           (frames, 3)
    LeftThigh/gyr.npy           (frames, 3)
    Pelvis/acc.npy              (frames, 3)
    Pelvis/gyr.npy              (frames, 3)
    RightAnkleAngle/angle.npy   (frames, 3)
    RightFoot/acc.npy           (frames, 3)
    RightFoot/gyr.npy           (frames, 3)
    RightHipAngle/angle.npy     (frames, 3)
    RightShank/acc.npy          (frames, 3)
    RightShank/gyr.npy          (frames, 3)
    RightThigh/acc.npy          (frames, 3)
    RightThigh/gyr.npy          (frames, 3)

After loading all files, the script concatenates them along the last dimension,
producing an array of shape (frames, total_dims).

Finally, it saves the merged data to SUB01/merged_imu_data.npy
"""

import os
import numpy as np

# Folder names and the .npy files expected within each
FOLDER_TO_NPYS = [
    ("LeftAnkleAngle", ["angle.npy"]),
    ("LeftFoot",       ["acc.npy", "gyr.npy"]),
    ("LeftHipAngle",   ["angle.npy"]),
    ("LeftShank",      ["acc.npy", "gyr.npy"]),
    ("LeftThigh",      ["acc.npy", "gyr.npy"]),
    ("Pelvis",         ["acc.npy", "gyr.npy"]),
    ("RightAnkleAngle",["angle.npy"]),
    ("RightFoot",      ["acc.npy", "gyr.npy"]),
    ("RightHipAngle",  ["angle.npy"]),
    ("RightShank",     ["acc.npy", "gyr.npy"]),
    ("RightThigh",     ["acc.npy", "gyr.npy"]),
]

def load_single_subject(subject_dir):
    """
    Load all .npy files specified in FOLDER_TO_NPYS from subject_dir.

    Each file is expected to have shape (frames, 3).
    All files must share the same number of frames.
    The resulting merged array has shape (frames, total_dims),
    concatenated along the last axis.
    """
    data_list = []
    frame_count = None

    for folder_name, npy_files in FOLDER_TO_NPYS:
        folder_path = os.path.join(subject_dir, folder_name)
        for fname in npy_files:
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                print(f"[WARNING] Missing file: {fpath}. Check data structure.")
                return None
            try:
                arr = np.load(fpath, allow_pickle=True)
            except Exception as e:
                print(f"[ERROR] Failed to load {fpath}: {e}")
                return None

            # Check that the array is (frames, 3)
            if arr.ndim != 2 or arr.shape[1] != 3:
                print(f"[WARNING] {fpath} has shape={arr.shape}, expected (frames, 3).")
                return None

            # Confirm consistent frame count
            if frame_count is None:
                frame_count = arr.shape[0]
            elif arr.shape[0] != frame_count:
                print(f"[WARNING] {fpath} has frame count {arr.shape[0]}, "
                      f"but previous files had {frame_count}.")
                return None

            data_list.append(arr)

    # Concatenate along the last axis => (frames, total_dims)
    merged = np.concatenate(data_list, axis=-1)
    return merged

if __name__ == "__main__":
    root_path = "./SUB01"  # Adjust this path as needed
    print(f"[INFO] Loading IMU data from: {root_path}")

    merged_data = load_single_subject(root_path)
    if merged_data is None:
        print("[ERROR] Merge failed: missing files or shape mismatch.")
    else:
        print(f"[INFO] Final shape: {merged_data.shape}")
        save_path = os.path.join(root_path, "merged_imu_data.npy")
        np.save(save_path, merged_data)
        print(f"[INFO] Saved merged data to: {save_path}")
