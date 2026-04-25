# ensemble csv for more files

- **Author:** atom1231
- **Votes:** 283
- **Ref:** atom1231/ensemble-csv-for-more-files
- **URL:** https://www.kaggle.com/code/atom1231/ensemble-csv-for-more-files
- **Last run:** 2025-06-16 16:07:02.130000

---

```python
* Supports more file ensembles compared to the previous version.
* Model weights are only exposed for public kernel originals.
```

```python
# 好的，這是一個非常實際的問題。當處理的檔案數量增加時，一次性將所有資料載入記憶體確實會很快導致「記憶體不足 (Out of Memory)」的錯誤。

# 為了解決這個問題，我將程式碼的邏輯進行了根本性的重構，改用一種 「串流處理 (Streaming)」 和 「記憶體映射 (Memory-Mapping)」 的方式。這種新方法不會一次把所有檔案都放進記憶體，因此可以處理遠超過10個，甚至更多的 submission 檔案，只受限於您的硬碟空間。

# 核心改善策略
# 計算平均值 (Mean) - 逐一累加法:

# 舊方法：先把所有檔案讀進記憶體，再一起計算。
# 新方法：建立一個空的「總和陣列」，然後一次只讀取一個檔案，將其數值乘上權重後，累加到「總和陣列」上。處理完一個檔案就將其從記憶體中釋放，再讀取下一個。這樣記憶體中最多只會有一個 submission 檔案的資料，非常節省空間。
# 計算中位數 (Median) - 記憶體映射檔案法:

# 計算中位數需要同時看到所有檔案的對應數值，這是最耗記憶體的部分。
# 新方法：我們利用 numpy.memmap 技術，在您的硬碟上建立一個「暫存工作區」(一個暫存檔案)。程式會逐一讀取 submission 檔案，並將其內容直接寫入這個硬碟上的工作區，而不是記憶體。當所有資料都寫入後，NumPy 可以直接在這個硬碟檔案上進行中位數計算，而不會佔用大量記憶體。計算完成後，這個暫存檔會被自動刪除。
# 完整 Python 程式碼 (記憶體優化版)
# 這是全新的、記憶體高效的函式。您可以放心用它來處理大量的檔案。

# Python

import pandas as pd
import numpy as np
import os
import gc # 引入記憶體回收模組

def ensemble_submissions_memory_efficient(file_paths, output_dir='.', weights=None):
    """
    使用記憶體優化策略，整合大量的 submission CSV 檔案。

    Args:
        file_paths (list): 一個包含所有 submission.csv 檔案路徑的列表。
        output_dir (str): 輸出結果檔案的儲存目錄。
        weights (list, optional): 一個與 file_paths 對應的權重列表。若為 None，則計算簡單平均。
    """
    num_files = len(file_paths)
    if num_files == 0:
        print("錯誤：檔案路徑列表是空的。")
        return

    # --- 權重驗證 ---
    is_weighted = weights is not None
    if is_weighted:
        if len(weights) != num_files:
            print(f"錯誤：權重列表的長度 ({len(weights)}) 與檔案列表的長度 ({num_files}) 不符。")
            return
        print(f"開始進行加權整合，使用的權重為: {weights}")
        total_weight = sum(weights)
    else:
        print(f"開始進行簡單平均整合（未提供權重）。")
        total_weight = num_files

    # --- 步驟 1: 讀取第一個檔案以取得基礎資訊 (索引、欄位、形狀) ---
    print("讀取第一個檔案以建立基礎架構...")
    try:
        first_df = pd.read_csv(file_paths[0], index_col='oid_ypos')
    except Exception as e:
        print(f"錯誤：讀取第一個檔案 {file_paths[0]} 失敗: {e}")
        return
        
    master_index = first_df.index
    master_columns = first_df.columns
    num_rows = len(master_index)
    num_cols = len(master_columns)
    del first_df # 立即釋放記憶體
    gc.collect()

    # --- 步驟 2: 使用「逐一累加法」計算平均值 ---
    print("\n--- 開始計算平均值 (記憶體優化模式) ---")
    # 建立一個全為 0 的陣列來儲存加總結果
    sum_array = np.zeros((num_rows, num_cols), dtype=np.float32)

    for i, file_path in enumerate(file_paths):
        print(f"處理中 (平均值) [{i+1}/{num_files}]: {file_path}")
        try:
            df = pd.read_csv(file_path, index_col='oid_ypos')
            df_reindexed = df.reindex(master_index)
            
            # 取得當前檔案的權重
            current_weight = weights[i] if is_weighted else 1.0
            
            # 將當前檔案的數值乘上權重後，累加到 sum_array
            sum_array += df_reindexed.values * current_weight
            
            del df, df_reindexed # 處理完畢，立即釋放記憶體
            gc.collect() # 主動觸發記憶體回收
        except Exception as e:
            print(f"警告：處理檔案 {file_path} 時發生錯誤，已跳過。錯誤訊息: {e}")
            continue

    # 計算最終平均值
    mean_data = sum_array / total_weight
    mean_output_filename = 'ensemble_weighted_mean.csv' if is_weighted else 'ensemble_mean.csv'
    df_mean = pd.DataFrame(mean_data, index=master_index, columns=master_columns)
    mean_output_path = os.path.join(output_dir, mean_output_filename)
    df_mean.to_csv(mean_output_path)
    print(f"平均值整合結果已儲存至: {mean_output_path}")
    del sum_array, df_mean, mean_data # 釋放記憶體
    gc.collect()

    #median
    # --- 步驟 3: 使用「記憶體映射檔案法」計算中位數 ---
    print("\n--- 開始計算中位數 (記憶體優化模式) ---")
    # 定義暫存檔案路徑
    temp_mmap_path = os.path.join(output_dir, 'median_temp_data.mmap')
    
    # 建立一個硬碟上的記憶體映射陣列
    # 使用 float32 可以節省一半的空間
    mmap_array = np.memmap(temp_mmap_path, dtype=np.float32, mode='w+', shape=(num_rows, num_cols, num_files))

    for i, file_path in enumerate(file_paths):
        print(f"處理中 (中位數) [{i+1}/{num_files}]: {file_path}")
        try:
            df = pd.read_csv(file_path, index_col='oid_ypos')
            df_reindexed = df.reindex(master_index)
            
            # 將資料直接寫入硬碟上的陣列，而不是記憶體
            mmap_array[:, :, i] = df_reindexed.values
            
            del df, df_reindexed
            gc.collect()
        except Exception as e:
            print(f"警告：處理檔案 {file_path} 時發生錯誤，已跳過。錯誤訊息: {e}")
            continue

    # NumPy 可以直接在 memmap 陣列上高效計算，不需載入全部資料
    print("正在從記憶體映射檔案計算中位數...")
    median_data = np.median(mmap_array, axis=2)
    
    # 將結果儲存
    df_median = pd.DataFrame(median_data, index=master_index, columns=master_columns)
    median_output_path = os.path.join(output_dir, 'submission_median.csv')
    df_median.to_csv(median_output_path)
    print(f"中位數整合結果已儲存至: {median_output_path}")

    # --- 步驟 4: 清理工作 ---
    del mmap_array, df_median, median_data # 確保物件被釋放
    gc.collect()
    try:
        os.remove(temp_mmap_path)
        print(f"已成功刪除暫存檔案: {temp_mmap_path}")
    except OSError as e:
        print(f"錯誤：無法刪除暫存檔案 {temp_mmap_path}。錯誤訊息: {e}")

# # --- 如何使用此函式 ---

# # 1. (示範用) 建立 10 個假的 submission 檔案
# print("-" * 40)
# num_demo_files = 10
# demo_dir = 'submissions_demo_large'
# print(f"正在建立 {num_demo_files} 個用於示範的假 submission 檔案...")
# if not os.path.exists(demo_dir):
#     os.makedirs(demo_dir)

# # 為了讓示範檔案大小更真實，可以增加資料量
# # 請注意：oid_count * ypos_count = 總列數。這裡設定為 40000 列。
# oid_count = 400
# ypos_count = 100
# x_cols = [f'x_{i}' for i in range(1, 70, 2)]
# oids = [f'id_{i:04d}' for i in range(oid_count)]
# all_keys = [f"{oid}_y_{y_pos}" for oid in oids for y_pos in range(ypos_count)]

# for i in range(num_demo_files):
#     dummy_df = pd.DataFrame(np.random.rand(len(all_keys), len(x_cols)), columns=x_cols, dtype=np.float32)
#     dummy_df['oid_ypos'] = all_keys
#     dummy_df = dummy_df.sample(frac=1).reset_index(drop=True)
#     dummy_df.to_csv(f'{demo_dir}/submission_{i}.csv', index=False)
#     print(f"已建立檔案: {demo_dir}/submission_{i}.csv")

# print(f"示範檔案已建立於 '{demo_dir}/' 資料夾中。")
# print("-" * 40)


# # 2. 定義您的檔案路徑列表和權重
# submission_files_large = [f'{demo_dir}/submission_{i}.csv' for i in range(num_demo_files)]
submission_files_large = [
    "/kaggle/input/caformrt-28-8/submission.csv",
    "/kaggle/input/convnext-base-29-6/submission.csv",
    #"/kaggle/input/convnext-base-lb31-data/submission.csv" ,   
    #"/kaggle/input/convnext0small-33-2/submission.csv",
    "/kaggle/input/convnext-huge-29-6/submission.csv",
    "/kaggle/input/caformer-m36-17-4overfit-28-2/submission.csv"
]



# 給予 10 個檔案不同的權重 (總和為 1)
#file_weights_large = [0.05, 0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05]
#file_weights_large = [0.6, 0.2, 0.1, 0.1] #027.5
#file_weights_large = [0.5, 0.15, 0.1, 0.1,0.15] #027.1
#file_weights_large = [0.5, 0.2, 0.05, 0.05,0.2] #
#file_weights_large = [0.4, 0.25, 0.05, 0.05,0.25] #
#file_weights_large = [0.4, 0.3, 0.3] #
file_weights_large = [0.4, 0.1, 0.1,0.4] #

# 3. 執行全新的記憶體優化整合函式
ensemble_submissions_memory_efficient(submission_files_large, weights=file_weights_large)
#no weight avg
#ensemble_submissions_memory_efficient(submission_files_large)
```

```python
# import pandas as pd
# import numpy as np
# import os

# def ensemble_submissions(file_paths, output_dir='.', weights=None):
#     """
#     將多個 submission CSV 檔案進行整合，可選擇計算加權平均。

#     Args:
#         file_paths (list): 一個包含所有 submission.csv 檔案路徑的列表。
#         output_dir (str): 輸出結果檔案的儲存目錄。
#         weights (list, optional): 一個與 file_paths 對應的權重列表。
#                                   若為 None，則計算簡單平均。預設為 None。
#     """
#     if not file_paths:
#         print("錯誤：檔案路徑列表是空的。")
#         return

#     # --- 新增：權重驗證 ---
#     if weights is not None:
#         if len(weights) != len(file_paths):
#             print(f"錯誤：權重列表的長度 ({len(weights)}) 與檔案列表的長度 ({len(file_paths)}) 不符。")
#             return
#         print(f"開始進行加權整合，使用的權重為: {weights}")
#     else:
#         print(f"開始進行簡單平均整合（未提供權重）。")

#     # --- 步驟 1: 讀取所有 CSV 檔案，並將 'oid_ypos' 設為索引 ---
#     try:
#         print("正在讀取 CSV 檔案...")
#         dataframes = [pd.read_csv(f, index_col='oid_ypos') for f in file_paths]
#     except FileNotFoundError as e:
#         print(f"錯誤：找不到檔案 -> {e}。請檢查您的檔案路徑是否正確。")
#         return
#     except KeyError:
#         print("錯誤：某個檔案中找不到 'oid_ypos' 欄位。請確保所有 CSV 都有此欄位。")
#         return

#     # --- 步驟 2: 資料對齊與堆疊 ---
#     print("正在對齊資料...")
#     master_index = dataframes[0].index
#     master_columns = dataframes[0].columns
#     aligned_data = [df.reindex(master_index).values for df in dataframes]
#     data_array = np.stack(aligned_data, axis=0).transpose(1, 2, 0)
#     print("資料堆疊完成。 陣列維度 (列數, 欄數, 檔案數):", data_array.shape)

#     # --- 步驟 3: 計算平均值 (加權或簡單) 與中位數 ---
    
#     # 計算平均值
#     if weights is not None:
#         print("正在計算加權平均值...")
#         # 使用 np.average 進行加權計算
#         mean_data = np.average(data_array, axis=2, weights=weights)
#         mean_output_filename = 'ensemble_weighted_mean.csv'
#     else:
#         print("正在計算簡單平均值...")
#         mean_data = np.mean(data_array, axis=2)
#         mean_output_filename = 'ensemble_mean.csv'
        
#     # 計算中位數 (中位數沒有加權的概念，維持原樣)
#     print("正在計算中位數...")
#     median_data = np.median(data_array, axis=2)

#     # --- 步驟 4: 建立結果 DataFrame 並儲存為 CSV 檔案 ---
#     # # 平均值結果
#     # df_mean = pd.DataFrame(mean_data, index=master_index, columns=master_columns)
#     # mean_output_path = os.path.join(output_dir, 'submission.csv')
#     # df_mean.to_csv(mean_output_path)
#     # print(f"平均值整合結果已儲存至: {mean_output_path}")

#     # 中位數結果
#     df_median = pd.DataFrame(median_data, index=master_index, columns=master_columns)
#     median_output_path = os.path.join(output_dir, 'submission.csv')
#     df_median.to_csv(median_output_path)
#     print(f"中位數整合結果已儲存至: {median_output_path}")

# # # --- 如何使用此函式 ---

# # # 1. (此為示範) 首先，建立一些假的 submission 檔案來模擬您的情況。
# # #    您可以跳過這部分，直接使用您自己的真實檔案。
# # print("正在建立用於示範的假 submission 檔案...")
# # if not os.path.exists('submissions_demo'):
# #     os.makedirs('submissions_demo')

# # # 從您的程式碼中取得欄位名稱 ('x_1', 'x_3', ..., 'x_69')
# # x_cols = [f'x_{i}' for i in range(1, 70, 2)]
# # oids = [f'id_{i:04d}' for i in range(5)]
# # all_keys = [f"{oid}_y_{y_pos}" for oid in oids for y_pos in range(70)]

# # # 建立 3 個假的 submission 檔案
# # for i in range(3): 
# #     dummy_df = pd.DataFrame(np.random.rand(len(all_keys), len(x_cols)), columns=x_cols)
# #     dummy_df['oid_ypos'] = all_keys
# #     # 將 DataFrame 的順序打亂，以模擬真實情況下順序不同的問題
# #     dummy_df = dummy_df.sample(frac=1).reset_index(drop=True)
# #     dummy_df.to_csv(f'submissions_demo/submission_{i}.csv', index=False)
# # print("示範檔案已建立於 'submissions_demo/' 資料夾中。")
# # print("-" * 40)


# # 2. 定義您所有 submission.csv 檔案的路徑列表。
# #    請將下面的列表換成您自己的檔案路徑。
# submission_files = [
#     "/kaggle/input/caformrt-28-8/submission.csv",
#     "/kaggle/input/convnext0small-33-2/submission.csv",
#     "/kaggle/input/convnext-base-29-6/submission.csv",
#     "/kaggle/input/convnext-base-lb31-data/submission.csv"
# ]
# # 例如:
# # submission_files = [
# #     'C:/my_models/model_A/submission.csv',
# #     'C:/my_models/model_B/submission.csv',
# #     'C:/my_models/model_C/submission.csv'
# # ]

# #file_weights = [0.8, 0.2]
# # 3. 執行整合函式。
# #ensemble_submissions(submission_files, weights=file_weights)
# #simple average
# ensemble_submissions(submission_files)
```

## Find files to load and create Dataset