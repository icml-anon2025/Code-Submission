import pandas as pd
import numpy as np
import pickle as pickle
import torch
from re import X
import sklearn as skl
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import utils as ut
import time


start_time = time.time()
data_waterpot = pd.read_csv("./TabCSDI/data_waterpot_3/df_train_waterpot_original.csv", header=0)
data_waterpot.replace("?", np.nan, inplace=True)

# data_waterpot_test = pd.read_csv("./original data/water/df_test_waterpot_before.csv", header=0).iloc[:, :]
# data_waterpot_test.replace("?", np.nan, inplace=True)

data_waterpot_test_after = pd.read_csv("./TabCSDI/data_waterpot_3/df_test_waterpot_before.csv", header=0).iloc[:, :]
data_waterpot_test_after = data_waterpot_test_after.dropna()

normalization_waterpot = './TabCSDI/data_waterpot_3/normalization_params.npy'
max, min = np.load(normalization_waterpot)


#train
file_path_train = './TabCSDI/save/waterpot_2_train_testing_fold5/generated_outputs_nsample100.pk'
processed_data = ut.load_and_process_data(file_path_train)

# Access elements
all_samples = processed_data['samples']

median_df = pd.DataFrame(np.median(all_samples, axis=1).squeeze(axis=2))
median_data_sample = np.median(all_samples, axis=1)
median_data_squeeze = median_data_sample.squeeze(axis=2)
median_df = pd.DataFrame(median_data_squeeze)

norm_waterpot_path = './TabCSDI/data_waterpot_3/normalization_params.npy'
descaled_median_df = ut.descale_median_data(median_df, norm_waterpot_path)

data_waterpot_filled = ut.fill_missing_values(data_waterpot, median_df, descaled_median_df)

data_waterpot_filled


#test
# file_path_test = './original data/water/generated_outputs_nsample100_water_test.pk'
# processed_data = ut.load_and_process_data(file_path_test)

# Access elements
# all_samples = processed_data['samples']

# median_data_sample = np.median(all_samples, axis=1)
# median_data_squeeze = median_data_sample.squeeze(axis=2)
# median_df = pd.DataFrame(median_data_squeeze)

# descaled_median_df = ut.descale_median_data(median_df, norm_waterpot_path)

# data_waterpot_test_filled = ut.fill_missing_values(data_waterpot_test, median_df, descaled_median_df)
# data_waterpot_test_filled



X_waterpot_train = data_waterpot_filled.iloc[:, :-1]
Y_waterpot_train = data_waterpot_filled.iloc[:, -1] 
# X_waterpot_test_before = data_waterpot_test_filled.iloc[:, :-1]
# Y_waterpot_test_before = data_waterpot_test_filled.iloc[:, -1] 
X_waterpot_test_after = data_waterpot_test_after.iloc[:, :-1]
Y_waterpot_test_after = data_waterpot_test_after.iloc[:, -1] 


model_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=10000, tol=1e-5)

model_svm.fit(X_waterpot_train, Y_waterpot_train)

y_train_pred = model_svm.predict(X_waterpot_train)
train_accuracy = accuracy_score(Y_waterpot_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# y_test_pred = model_svm.predict(X_waterpot_test_before)
# test_accuracy = accuracy_score(Y_waterpot_test_before, y_test_pred)
# print(f"Test Accuracy (Before): {test_accuracy:.4f}")

y_test_pred_after = model_svm.predict(X_waterpot_test_after)
test_accuracy_after = accuracy_score(Y_waterpot_test_after, y_test_pred_after)
print(f"Test Accuracy (After): {test_accuracy_after:.4f}")

# process_time = time.time() - start_time
# print(f"Process Time: {process_time:.4f} seconds")