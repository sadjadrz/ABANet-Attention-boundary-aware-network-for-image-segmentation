import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

#import matplotlib.pyplot as plt
import torch
import numpy as np

import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
from sklearn.metrics import accuracy_score , f1_score , jaccard_score ,precision_score , recall_score
from glob import glob
from tqdm import tqdm
import pandas as pd
import segmentation_models_pytorch as smp

def accuracy_f1score_cal(ground_truth_path, predicted_path):
    ground_truth_label = sorted(glob(ground_truth_path))
    predict_label = sorted(glob(predicted_path))

    SCORE = []
    for truth, predict in tqdm(zip(ground_truth_label, predict_label), total=len(ground_truth_label)):
        image_name = truth.split("/")[-1]
        truth = cv2.imread(truth,cv2.IMREAD_UNCHANGED)
        #truth = cv2.resize(truth, (256, 256))
        _,truth = cv2.threshold(truth, 127, 255, cv2.THRESH_BINARY)
        truth = truth/255
        truth = truth.astype(np.int32)

        predict = cv2.imread(predict,cv2.IMREAD_UNCHANGED)
        predict = cv2.resize(predict, (250, 250))
        predict = predict[:,:,0]
        _, predict = cv2.threshold(predict, 127, 255, cv2.THRESH_BINARY)
        predict = predict / 255
        predict = predict.astype(np.int32)
        truth = truth.flatten()
        predict = predict.flatten()
        # cv2.imshow('d',predict)
        # cv2.waitKey(0)
        acc_value = accuracy_score(truth, predict)
        f1 = f1_score(truth, predict, labels=[0, 1], average="binary")
        jaccard = jaccard_score(truth, predict, labels=[0, 1], average="binary")
        preceision = precision_score(truth, predict, labels=[0, 1], average="binary")
        recall = recall_score(truth, predict, labels=[0, 1], average="binary")

        SCORE.append([image_name, acc_value, f1, jaccard, preceision, recall])
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    df = pd.DataFrame(SCORE, columns=["imagename", "accuracy", "F1", "jaccard", "preceision", "recall"])

    df.to_csv("score_focal.csv")
    return score
def check_path(path):
    is_directory = False
    is_file = False
    is_other = False
    if os.path.isdir(path):
        is_directory = True
    elif os.path.isfile(path):
        is_file = True
    else:
        is_other = True

    return is_directory, is_file, is_other
def changeFileName(folderPath):
    is_directory, is_file, is_other = check_path(folderPath)
    if is_directory:
        path, dirs, files = os.walk(folderPath).__next__()
        file_count = len(files)
        dirs_count = len(dirs)

        # Process files in the directory if any
        for d in tqdm(dirs):
            dir_path = folderPath + "/" + d
            _, _, files = os.walk(dir_path).__next__()
            for f in files:
                image_path = dir_path + "/" + f
                #split_path = f.rsplit("_")
                #new_name = split_path[1]+f[-4:]
                new_name = f[:-3]+'jpg'
                new_image_path = dir_path +"/"+ new_name
                os.rename(image_path, new_image_path)
if __name__ == '__main__':
    #changeFileName('test_data/test_results')

    acc_erode = accuracy_f1score_cal("test_data/test_images/test_images/*","test_data/test_results/test_images/attention/8000/*")
    print(f"accuracy_result: {acc_erode[0]:0.5f}")
    print(f"f1_Score_result: {acc_erode[1]:0.5f}")
    print(f"IoU:result {acc_erode[2]:0.5f}")
    print(f"pre:result {acc_erode[3]:0.5f}")
    print(f"recall:result {acc_erode[4]:0.5f}")