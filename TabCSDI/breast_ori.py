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
data_breast_raw = pd.read_csv("./TabCSDI/data_breast_2/df_train_Breast_Cancer_original.csv", header=0)
data_breast = data_breast_raw.iloc[:,1:]
data_breast_id = data_breast_raw.iloc[:,0]
data_breast.replace("?", np.nan, inplace=True)

# data_breast_test_raw = pd.read_csv("./TabCSDI/data_breast_2/df_test_Breast_Cancer_before.csv", header=0)
# data_breast_test = data_breast_test_raw.iloc[:,1:]
# data_breast_test_id = data_breast_test_raw.iloc[:,0]
# data_breast_test.replace("?", np.nan, inplace=True)

data_breast_test_after = pd.read_csv("./TabCSDI/data_breast_2/df_test_Breast_Cancer_before.csv", header=0).iloc[:, :]
data_breast_test_after = data_breast_test_after.dropna()
#train

file_path_train = './TabCSDI/save/breast_2_train_testing_fold5/generated_outputs_nsample100.pk'
processed_data = ut.load_and_process_data(file_path_train)
all_samples = processed_data['samples']

median_df = pd.DataFrame(np.median(all_samples, axis=1).squeeze(axis=2))
norm_breast_path = './TabCSDI/data_breast_2/normalization_params.npy'
descaled_median_df = ut.descale_median_data(median_df, norm_breast_path)

data_breast_filled = ut.fill_missing_values(data_breast, median_df, descaled_median_df)

data_train_breast_complete = pd.concat([data_breast_id, data_breast_filled], axis=1)
data_train_breast_complete["0"] = data_train_breast_complete["0"].astype(int)

#test

# file_path_test = './original data/breast/generated_outputs_nsample100_breast_test.pk'
# # Access elements'
# processed_data = ut.load_and_process_data(file_path_test)
# all_samples = processed_data['samples']

# median_data_sample = np.median(all_samples, axis=1)
# median_data_squeeze = median_data_sample.squeeze(axis=2)
# median_df = pd.DataFrame(median_data_squeeze)

# descaled_median_test_df = ut.descale_median_data(median_df, norm_breast_path)

# data_breast_filled_test = ut.fill_missing_values(data_breast_test, median_df, descaled_median_df)
# data_test_breast_complete = pd.concat([data_breast_test_id, data_breast_filled_test], axis=1)
# data_test_breast_complete["0"] = data_test_breast_complete["0"].astype(int)

X_breast_train = data_train_breast_complete.iloc[:, :-1]  # Drop the last column
Y_breast_train = data_train_breast_complete.iloc[:, -1]   # Select the last column
# X_breast_test = data_test_breast_complete.iloc[:, :-1]  # Drop the last column
# Y_breast_test = data_test_breast_complete.iloc[:, -1] 
X_breast_test_after = data_breast_test_after.iloc[:, :-1] 
Y_breast_test_after = data_breast_test_after.iloc[:, -1] 

model_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=10000, tol=1e-5)
model_svm.fit(X_breast_train, Y_breast_train)

y_train_pred = model_svm.predict(X_breast_train)
train_accuracy = skl.metrics.accuracy_score(Y_breast_train, y_train_pred)
print("Accuracy:", train_accuracy)

# y_test_pred = model_svm.predict(X_breast_test)
# test_accuracy = skl.metrics.accuracy_score(Y_breast_test, y_test_pred)
# print("Accuracy:", test_accuracy)

y_test_pred_after = model_svm.predict(X_breast_test_after)
test_accuracy = skl.metrics.accuracy_score(Y_breast_test_after, y_test_pred_after)
print("Accuracy after:", test_accuracy)

# process_time = time.time() - start_time
# print(f"Process Time: {process_time:.4f} seconds")