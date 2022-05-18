import pandas as pd
import numpy as np

def extract_sliding_window(subject_df, window_size=150, stride=3):
    slide_window_data = None
    window_id = 0
    # subject data contains fNIRS for 16 tasks
    for task_id in range(0, 16):
        task_data = subject_df[int(426*task_id): int(426*(task_id+1))].copy().reset_index(drop=True)
        for i in range(199-(window_size-1), 424-(window_size-1)+1, 3):
            window_data = task_data[i: i+window_size].copy()
            window_data['window'] = window_id
            window_id += 1
            if slide_window_data is None:
                slide_window_data = window_data
            else:
                slide_window_data = pd.concat([slide_window_data, window_data], ignore_index=True)
    
    return slide_window_data

def data_loading_function(slide_window_data, feature_column=['AB_I_O', 'AB_PHI_O', 'AB_I_DO', 'AB_PHI_DO', 'CD_I_O', 'CD_PHI_O',
       'CD_I_DO', 'CD_PHI_DO'], num_windows=1216):
    instance_features, instance_labels = [], []
    slide_window_data = slide_window_data[feature_column + ['window'] + ['label']]
    for i in range(0, num_windows):
        features = slide_window_data.iloc[:, :-2].loc[slide_window_data['window']==i].values
        label = slide_window_data.loc[slide_window_data['window']==i].label.values[0]
        
        instance_features.append(features)
        instance_labels.append(label)
        
    instance_features = np.array(instance_features, dtype=np.float32)
    instance_labels = np.array(instance_labels, dtype=np.int64)
    return instance_features, instance_labels