import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_imu_data(filepath: str, time_column: tuple = ('Chest', 'Time', 'S')) -> pd.DataFrame:
    """
    Load IMU data with multi-index header, and extract the time column.
    Skips metadata row with 'S', 'X', 'Y', etc.
    """
    df = pd.read_csv(filepath, header=[1, 2, 3])
    df['Time'] = df[time_column].astype(float)
    return df

def load_angle_data(filepath: str) -> pd.DataFrame:
    """
    Load IMU data with multi-index header, and extract the time column.
    Skips metadata row with 'S', 'X', 'Y', etc.
    """
    df = pd.read_csv(filepath, header=[1, 2, 3], index_col=0)
    return df

def resample_full_data_with_time_gaps(df: pd.DataFrame, target_rate: int = 200, time_col='Time') -> pd.DataFrame:
    """
    Resample time series data even with time gaps. Each continuous block is interpolated independently.
    """
    interval = 1 / target_rate
    t_all = df[time_col].values
    time_diff = np.diff(t_all)

    # Threshold: if time difference > 2x interval, it's a break
    break_indices = np.where(time_diff > interval * 2)[0]
    split_indices = np.concatenate([[0], break_indices + 1, [len(t_all)]])

    resampled_blocks = []

    for i in range(len(split_indices) - 1):
        start, end = split_indices[i], split_indices[i + 1]
        segment = df.iloc[start:end].copy()

        t_segment = segment[time_col].values
        if len(t_segment) < 2:
            continue

        t_new = np.arange(t_segment[0], t_segment[-1], interval)
        segment_data = {time_col: t_new}

        for col in df.columns:
            if col == time_col:
                continue
            try:
                f = interp1d(t_segment, segment[col].values, kind='linear', fill_value="extrapolate")
                col_name = '_'.join(col) if isinstance(col, tuple) else col
                segment_data[col_name] = f(t_new)
            except Exception:
                continue

        resampled_blocks.append(pd.DataFrame(segment_data))

    return pd.concat(resampled_blocks, ignore_index=True)

def resample_full_data_from_index_reset(df: pd.DataFrame, target_rate: int = 200, index_col: str = None) -> pd.DataFrame:
    """
    Resample joint angle time series with index reset (e.g., first column resets to 0 when new segment starts).
    Interpolates each continuous segment independently.
    """
    interval = 1 / target_rate
    
    # Step 1: Detect breakpoints from index resets
    index_vals = df.iloc[:, 0].values  # First column as segment marker (e.g., 0,1,2,...)
    break_indices = [0]
    for i in range(1, len(index_vals)):
        if index_vals[i] == 0:
            break_indices.append(i)
    break_indices.append(len(df))  # Final boundary

    # Step 2: Interpolate each segment
    resampled_blocks = []

    for i in range(len(break_indices) - 1):
        start, end = break_indices[i], break_indices[i + 1]
        segment = df.iloc[start:end].copy()

        # Reconstruct virtual time (assume 100Hz original sampling)
        t_segment = np.arange(len(segment)) / 100
        t_new = np.arange(0, t_segment[-1], interval)
        segment_data = {'Time': t_new}

        for col in df.columns[0:]:  # Skip first column (segment index)
            try:
                f = interp1d(t_segment, segment[col].values, kind='linear', fill_value="extrapolate")
                col_name = '_'.join(col) if isinstance(col, tuple) else col
                segment_data[col_name] = f(t_new)
            except Exception as e:
                print(f"Skipping column {col} due to error: {e}")
                continue

        resampled_blocks.append(pd.DataFrame(segment_data))

    return pd.concat(resampled_blocks, ignore_index=True)



def plot_sensor_waveform(df: pd.DataFrame, colname: str, title: str = None):
    """
    Plot the waveform of a single sensor signal across the full timeline.
    """
    plt.figure(figsize=(30, 5))
    plt.plot(df['Time'], df[colname], linewidth=1.2)
    plt.title(title or f"Waveform of {colname}")
    plt.xlabel("Time (s)")
    plt.ylabel("Sensor Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sensor_waveform_comparison(csv_data: pd.DataFrame, npy_data: np.ndarray, time_csv: np.ndarray, time_npy: np.ndarray, col_prefix: str):
    # Make sure the lengths match by truncating to the shortest
    min_len = min(len(time_csv), len(time_npy), npy_data.shape[0], csv_data.shape[0])
    time_csv = time_csv[:min_len]
    time_npy = time_npy[:min_len]
    npy_data = npy_data[:min_len, :]
    #csv_data = csv_data.iloc[:min_len]
    csv_data = csv_data[:min_len, :]
    print(min_len)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    titles = ['X lfoot', 'Y lfoot', 'Z lfoot']
    axes = ['X', 'Y', 'Z']

    for i in range(3):
        csv_col = f"{col_prefix}_{axes[i]}"
        axs[i].plot(time_csv, csv_data[:, i], label='resampled CSV', alpha=0.7)  #csv_data[csv_col]
        axs[i].plot(time_npy, npy_data[:, i], label='npy ori-s100', alpha=0.7)
        axs[i].set_ylabel('lfoot')
        axs[i].set_title(titles[i])
        axs[i].legend()

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()



'''
# === Test ===
file_path = "/Users/ccy/Documents/CMU/Spring2025/42696 Wearable Health Technology/Projects/wearable-project/data preprocessing/Data/IMUExerciseClassification/parsed_joint_angles/SUB01/Run/Merged_Run.csv"

# Load and resample
df_raw = load_angle_data(file_path)
df_resampled = resample_full_data_from_index_reset(df_raw, target_rate=200, index_col=0)
df_resampled.to_csv("resampled_no_segments.csv", index=None)
print(df_resampled)

#df_resampled = pd.read_csv(file_path)
# Choose a column to visualize (adjust index as needed)
target_column = 'LeftAnkleAngle_ML_X'
plot_sensor_waveform(df_resampled, target_column)
'''


