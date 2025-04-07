import pandas as pd
import numpy as np
import pickle as pickle
import torch
from re import X
import sklearn as skl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import utils as ut


data_air_raw = pd.read_csv("./TabCSDI/data_airquality_2/train_air_quality_original.csv", header=0)
data_air = data_air_raw
data_air.replace("?", np.nan, inplace=True)

data_air_test_raw = pd.read_csv("./TabCSDI/data_airquality_2/test_air_quality_original.csv", header=0)
data_air_test = data_air_test_raw
data_air_test.replace("?", np.nan, inplace=True)

data_air_test_after = data_air_test.dropna()


file_path_train = './TabCSDI/save/airquality_2_train_testing_fold5/generated_outputs_nsample100.pk'
processed_data = ut.load_and_process_data(file_path_train)
all_samples = processed_data['samples']

median_df = pd.DataFrame(np.median(all_samples, axis=1).squeeze(axis=2))
norm_air_path = './TabCSDI/data_airquality_2/normalization_params.npy'
descaled_median_df = ut.descale_median_data(median_df, norm_air_path)


if descaled_median_df.index.equals(data_air.index):
    print("Indexes are identical and in the same order")
else:
    print("Indexes are different or not in the same order")


data_train_air_complete = ut.fill_missing_values(data_air, median_df, descaled_median_df)


# file_path_test = './TabCSDI/original data/air/generated_outputs_nsample100_air_test.pk'
# processed_data = ut.load_and_process_data(file_path_test)

# # Access elements
# all_samples = processed_data['samples']

# median_data_sample = np.median(all_samples, axis=1)
# median_data_squeeze = median_data_sample.squeeze(axis=2)
# median_df = pd.DataFrame(median_data_squeeze)

# descaled_median_test_df = ut.descale_median_data(median_df, norm_air_path)


# data_test_air_complete = ut.fill_missing_values(data_air_test, median_df, descaled_median_test_df)
# data_test_air_complete = pd.concat([data_air_test_id, data_air_filled_test], axis=1)
# data_test_air_complete["0"] = data_test_air_complete["0"].astype(int)

X_air_train = data_train_air_complete.drop(columns=["T"])
Y_air_train = data_train_air_complete["T"]
# X_air_test = data_test_air_complete.drop(columns=["T"])
# Y_air_test = data_test_air_complete["T"]
X_air_test_after = data_air_test_after.drop(columns=["T"])
Y_air_test_after = data_air_test_after["T"]



clf = LinearRegression(fit_intercept = False).fit(X_air_train,Y_air_train)
# y_pred = clf.predict(X_test)
# score = mean_squared_error(y_test, y_pred)

Y_air_pred_train = clf.predict(X_air_train)  # or use X_air_test if evaluating on test data
mse_train = mean_squared_error(Y_air_train, Y_air_pred_train)
print("MSE Training:", mse_train)

# Y_air_pred_test = clf.predict(X_air_test)  # or use X_air_test if evaluating on test data
# mse_test = mean_squared_error(Y_air_test, Y_air_pred_test)
# print("MSE Test:", mse_test)


Y_air_pred_test_after = clf.predict(X_air_test_after)  # or use X_air_test if evaluating on test data
mse_test_after = mean_squared_error(Y_air_test_after, Y_air_pred_test_after)
print("MSE After:", mse_test_after)


# y_test_pred = model_svm.predict(X_air_test)
# test_accuracy = skl.metrics.accuracy_score(Y_air_test, y_test_pred)
# print("Accuracy:", test_accuracy)

# y_test_pred_after = model_svm.predict(X_air_test_after)
# test_accuracy = skl.metrics.accuracy_score(Y_air_test_after, y_test_pred_after)
# print("Accuracy after:", test_accuracy)