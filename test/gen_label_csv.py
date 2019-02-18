import os
import math
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm

label_warp = {'OK': 0,
              'colorbeam': 1,
              'ghost': 2,
              'innerDust': 3,
              'largerLight': 4,
              'longbeam': 5,
              'multibeam': 6,
              'shortbeam': 7,
              }

# train data
data_path = 'medadata/2nd/temp/'
img_path, label = [], []

for first_path in os.listdir(data_path):
    first_path = osp.join(data_path, first_path)
    if 'OK' in first_path:
        for img in os.listdir(first_path):
            img_path.append(osp.join(first_path, img))
            label.append('OK')
    else:
        for second_path in os.listdir(first_path):
            defect_label = second_path
            second_path = osp.join(first_path, second_path)
            if defect_label != '其他':
                for img in os.listdir(second_path):
                    img_path.append(osp.join(second_path, img))
                    label.append(defect_label)
            else:
                for third_path in os.listdir(second_path):
                    third_path = osp.join(second_path, third_path)
                    if osp.isdir(third_path):
                        for img in os.listdir(third_path):
                            if 'DS_Store' not in img:
                                img_path.append(osp.join(third_path, img))
                                label.append(defect_label)

label_file = pd.DataFrame({'Id': img_path, 'Target': label})
label_file['Target'] = label_file['Target'].map(label_warp)

label_file.to_csv('data/label.csv', index=False)

# test data
# test_data_path = 'data/guangdong_round1_test_a_20180916'
# all_test_img = os.listdir(test_data_path)
# test_img_path = []

# for img in all_test_img:
#     if osp.splitext(img)[1] == '.jpg':
#         test_img_path.append(osp.join(test_data_path, img))

# test_file = pd.DataFrame({'img_path': test_img_path})
# test_file.to_csv('data/test.csv', index=False)
