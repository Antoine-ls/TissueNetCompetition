import glob
import cv2
import numpy as np
import json
import pandas as pd
import re
import time
from tqdm import tqdm

file_path_list = glob.glob("../data/downsampled_images/*")
file_path_list.sort()

def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m"%'   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)


# pattern = re.compile(r'[^/]*\.jpeg')

file_name_list = []
for file_path in file_path_list:
    file_name = file_path.split('/')[-1].split('.')[0] + '.tif'
    file_name_list.append(file_name)


file_name_df = pd.DataFrame({'filename': file_name_list, 'filepath': file_path_list})

res_np = np.empty(shape=(len(file_name_list), 12))
for idx, (file_path, per) in enumerate(zip(file_path_list, tqdm(range(len(file_name_list))))):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_avg, h_var = np.average(img[0]), np.var(img[0])
    h_max, h_min = np.max(img[0]), np.min(img[0])

    s_avg, s_var = np.average(img[1]), np.var(img[1])
    s_max, s_min = np.max(img[1]), np.min(img[1])

    v_avg, v_var = np.average(img[2]), np.var(img[2])
    v_max, v_min = np.max(img[2]), np.min(img[2])

    res_np[idx] = [h_avg, h_var, h_max, h_min, s_avg, s_var, s_max, s_min, v_avg, v_var, v_max, v_min]

res_df = pd.DataFrame(res_np, columns=["h_avg", "h_var", "h_max", "h_min", "s_avg", "s_var", "s_max", "s_min", "v_avg", "v_var", "v_max", "v_min"])

res = pd.concat([file_name_df, res_df], axis=1)
res.to_csv("statistic_images_{}.csv".format(len(file_path_list)))
print("Statistic saved")