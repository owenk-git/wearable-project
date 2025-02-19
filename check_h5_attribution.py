import h5py

h5path = "Data/2_Processed/walking_test_data.h5"  # 替換成你的 HDF5 檔案

with h5py.File(h5path, 'r') as fh:
    subs = list(fh.keys())  # 列出所有 subjects
    print(f"Subjects found: {len(subs)}")
    
    # 檢查前 5 個 subjects
    for sub in subs[:100]:
        print(f"Checking {sub} attributes: {list(fh[sub].attrs.keys())}")  # 列出屬性

        # 檢查前 10 個 subjects 並打印 `checks_passed` 值
    for sub in subs[:10]:
        checks_passed = fh[sub].attrs.get('checks_passed', 'Attribute not found')
        print(f"Subject: {sub}, checks_passed: {checks_passed}")

