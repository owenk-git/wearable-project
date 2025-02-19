import h5py
import numpy as np

def create_test_h5(original_h5, test_h5, max_subjects=100):
    with h5py.File(original_h5, 'r') as orig_f:
        with h5py.File(test_h5, 'w') as new_f:
            subs = list(orig_f.keys())

            # 只選擇 s0-s100
            selected_subs = [sub for sub in subs if sub.startswith("s") and 0 <= int(sub[1:]) <= max_subjects]

            for sub in selected_subs:
                orig_f.copy(sub, new_f)

            print(f"✅ 測試版 {test_h5} 建立成功，共 {len(selected_subs)} 個 subjects")

# 執行測試版建立（請確保 `walking_meta.h5` 存在）
create_test_h5('Data/1_Extracted/running_meta.h5', 'Data/1_Extracted/walking_test_meta.h5', max_subjects=300)
