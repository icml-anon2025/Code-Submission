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
data_online= pd.read_csv("./TabCSDI/data_online_ed_2/df_train_online_ed_original.csv", header = 0)
data_online.replace("?", np.nan, inplace=True)


# data_online_test = pd.read_csv("./original data/online/df_test_online_ed_before.csv", header = 0)
# data_online_test.replace("?", np.nan, inplace=True)


data_online_test_after = pd.read_csv("./TabCSDI/data_online_ed_2/df_test_online_ed_before.csv", header = 0)
data_online_test_after = data_online_test_after.dropna()

norm_online_path = "./TabCSDI/data_online_ed_2/normalization_params.npy"


train_detoken_path = './TabCSDI/save/online_ed_fold5/detoken_samples_output_online_ed_2.pk'
median_df = ut.detoken_load_and_compute_median(train_detoken_path)
categorical = median_df.drop(columns=[0,1,2])
numerical = median_df.drop(columns=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

categorical_adjust = categorical-1

descaled_numerical_median_df = ut.descale_median_data(numerical, norm_online_path)
combine_descaled_median_df = pd.concat([descaled_numerical_median_df, categorical_adjust], axis=1)
data_train_online_complete = ut.fill_missing_values(data_online, median_df, combine_descaled_median_df)



# test_detoken_path = './original data/online/test_detoken_samples_output_online_ed_2_20250220_035448.pk'
# median_test_df = ut.detoken_load_and_compute_median(test_detoken_path)
# categorical_test = median_test_df.drop(columns=[0,1,2])
# numerical_test = median_test_df.drop(columns=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

# categorical_test_adjust = categorical_test-1

# descaled_numerical_median_test_df = ut.descale_median_data(numerical_test, norm_online_path)
# combine_descaled_median_test_df = pd.concat([descaled_numerical_median_test_df, categorical_test_adjust], axis=1)
# data_test_online_complete = ut.fill_missing_values(data_online_test, median_test_df, combine_descaled_median_test_df)





X_breast_train = data_train_online_complete.iloc[:, :-1]  # Drop the last column
Y_breast_train = data_train_online_complete.iloc[:, -1] 
# X_breast_test = data_test_online_complete.iloc[:, :-1]  # Drop the last column
# Y_breast_test = data_test_online_complete.iloc[:, -1] 
X_breast_test_after = data_online_test_after.iloc[:, :-1]  # Drop the last column
Y_breast_test_after = data_online_test_after.iloc[:, -1] 


model_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=13, max_iter=10000, tol=1e-5)
model_svm.fit(X_breast_train, Y_breast_train)

y_train_pred = model_svm.predict(X_breast_train)
train_accuracy = skl.metrics.accuracy_score(Y_breast_train, y_train_pred)
print("Accuracy train:", train_accuracy)


# y_test_pred = model_svm.predict(X_breast_test)
# test_accuracy = skl.metrics.accuracy_score(Y_breast_test, y_test_pred)
# print("Accuracy test:", test_accuracy)


y_test_pred_after = model_svm.predict(X_breast_test_after)
test_after_accuracy = skl.metrics.accuracy_score(Y_breast_test_after, y_test_pred_after)
print("Accuracy test after:", test_after_accuracy)

# process_time = time.time() - start_time
# print(f"Process Time: {process_time:.4f} seconds")

# norm_param = np.load(norm_online_path)
# max_value, min_value = norm_param[0], norm_param[1]

# print(max_value)
# print(min_value)


