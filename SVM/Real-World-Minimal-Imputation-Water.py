import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import os
from sklearn.impute import SimpleImputer, KNNImputer
# Generate random repairs and replace with edge repairs if available
from hyperimpute.plugins.imputers import Imputers
from sklearn.metrics import f1_score

import time
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
def translate_indices(globali, imap):
    lset = set(globali)
    return [s for s,t in enumerate(imap) if t in lset]

def error_classifier(total_labels, full_data):
    indices = [i[0] for i in total_labels]
    labels = [int(i[1]) for i in total_labels]
    if np.sum(labels) < len(labels):
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=200, fit_intercept=True)
        clf.fit(full_data[indices,:],labels)
        return clf
    else:
        return None

def ec_filter(dirtyex, full_data, clf, t=0.90):
    if clf != None:
        pred = clf.predict_proba(full_data[dirtyex,:])
        return [j for i,j in enumerate(dirtyex) if pred[i][0] < t]

    #print("CLF none")
    return dirtyex




def activeclean(dirty_data, clean_data, test_data, full_data, indextuple, task='classification', batchsize=50, total=10000):
    # makes sure the initialization uses the training data
    X = dirty_data[0][translate_indices(indextuple[0], indextuple[1]), :]
    y = dirty_data[1][translate_indices(indextuple[0], indextuple[1])]

    X_clean = clean_data[0]
    y_clean = clean_data[1]

    X_test = test_data[0].values
    y_test = test_data[1].values

    # print("[ActiveClean Real] Initialization")

    lset = set(indextuple[2])
    dirtyex = [i for i in indextuple[0]]
    cleanex = []

    total_labels = []
    total_cleaning = 0  # Initialize the total count of missing or originally dirty examples

    ##Not in the paper but this initialization seems to work better, do a smarter initialization than
    ##just random sampling (use random initialization)
    topbatch = np.random.choice(range(0, len(dirtyex)), batchsize)
    examples_real = [dirtyex[j] for j in topbatch]
    examples_map = translate_indices(examples_real, indextuple[2])

    # Apply Cleaning to the Initial Batch

    cleanex.extend(examples_map)
    for j in set(examples_real):
        dirtyex.remove(j)

    if task =='classification':
        clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
    else:
        clf = SGDRegressor(penalty=None,max_iter=200)
    clf.fit(X_clean[cleanex, :], y_clean[cleanex])

    for i in range(50, total, batchsize):
        # print("[ActiveClean Real] Number Cleaned So Far ", len(cleanex))
        ypred = clf.predict(X_test)
        # print("[ActiveClean Real] Prediction Freqs",np.sum(ypred), np.shape(ypred))
        # print(f"[ActiveClean Real] Prediction Freqs sum ypred: {np.sum(ypred)} shape of ypred: {np.shape(ypred)}")
        # print(classification_report(y_test, ypred))

        # Sample a new batch of data
        examples_real = np.random.choice(dirtyex, batchsize)

        # Calculate the count of missing or originally dirty examples within the batch
        missing_count = sum(1 for r in examples_real if r in indextuple[1])
        total_cleaning += missing_count  # Add the count to the running total

        examples_map = translate_indices(examples_real, indextuple[2])

        total_labels.extend([(r, (r in lset)) for r in examples_real])

        # on prev. cleaned data train error classifier
        ec = error_classifier(total_labels, full_data)

        for j in examples_real:
            try:
                dirtyex.remove(j)
            except ValueError:
                pass

        dirtyex = ec_filter(dirtyex, full_data, ec)

        # Add Clean Data to The Dataset
        cleanex.extend(examples_map)

        # uses partial fit (not in the paper--not exactly SGD)
        clf.partial_fit(X_clean[cleanex, :], y_clean[cleanex])

        print('Clean', len(cleanex))
        # print("[ActiveClean Real] Accuracy ", i ,accuracy_score(y_test, ypred,normalize = True))
        #print(f"[ActiveClean Real] Iteration: {i} Accuracy: {accuracy_score(y_test, ypred,normalize = True)}")
        if len(dirtyex) < 50:
            print("[ActiveClean Real] No More Dirty Data Detected")
            print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
            # return total_cleaning, 0 if clf.score(X_clean[cleanex, :], y_clean[cleanex]) is None else clf.score(
            #     X_clean[cleanex, :], y_clean[cleanex])cleanex
            if task == 'classification':
                ypred = clf.predict(X_test)
                return total_cleaning, 0 if accuracy_score(y_test, ypred) is None else accuracy_score(y_test, ypred)

            else:
                ypred = clf.predict(X_test)
                return total_cleaning, 0 if mean_squared_error(y_test, ypred) is None else mean_squared_error(y_test, ypred)

    print("[ActiveClean Real] Total Dirty records cleaned", total_cleaning)
    if task == 'classification':
        ypred = clf.predict(X_test)
        return total_cleaning, 0 if accuracy_score(y_test, ypred) is None else accuracy_score(y_test, ypred)

    else:
        ypred = clf.predict(X_test)
        return total_cleaning, 0 if mean_squared_error(y_test, ypred) is None else mean_squared_error(y_test, ypred)



def generate_random_repair_with_edge(dataset, index_and_edge_repair, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    col_min = np.nanmin(dataset, axis=0)
    col_max = np.nanmax(dataset, axis=0)

    new_dataset = dataset.copy()
    for key, edge_repair in index_and_edge_repair.items():
        new_dataset[key] = edge_repair

    nan_mask = np.isnan(new_dataset)

    random_repaired_dataset = new_dataset.copy()
    for i in range(new_dataset.shape[1]):
        random_repaired_dataset[nan_mask[:, i], i] = np.random.uniform(col_min[i], col_max[i], size=nan_mask[:, i].sum())

    return random_repaired_dataset

def findEdgeRepair(incomplete_example, repaired_other_examples, model, y_incompleteExample):
    col_min = np.nanmin(repaired_other_examples, axis=0)
    col_max = np.nanmax(repaired_other_examples, axis=0)
    repaired_example = incomplete_example.copy()

    for i, val in enumerate(incomplete_example):
        if np.isnan(val):
            coef_value = model.coef_[0][i]
            if y_incompleteExample > 0:
                repaired_example[i] = col_min[i] if coef_value > 0 else col_max[i]
            else:
                repaired_example[i] = col_max[i] if coef_value > 0 else col_min[i]

    return repaired_example

def checkSV(incompleteExample, repairedOtherExamples, labels, y_incompleteExample, seed=None):
    model = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed, warm_start=True)
    model.fit(repairedOtherExamples, labels)

    repaired_example = findEdgeRepair(incompleteExample, repairedOtherExamples, model, y_incompleteExample)

    decision_value = np.dot(model.coef_, repaired_example) + model.intercept_
    product = y_incompleteExample * decision_value

    if product < 1:
        return True, repaired_example

    return False, repaired_example

def checkImputationNecessity(possible_repaired_dataset, original_dataset, labels, example_index, seed=None):
    possible_repaired_data_subset = np.delete(possible_repaired_dataset, example_index, axis=0)
    labels_subset = np.delete(labels, example_index, axis=0)
    if possible_repaired_data_subset.shape[0] == 0 or len(np.unique(labels_subset)) < 2:
        raise ValueError("Insufficient data or classes")

    incomplete_example = original_dataset[example_index]
    return checkSV(incomplete_example, possible_repaired_data_subset, labels_subset, labels[example_index], seed)

def findminimalImputation(original_dataset, labels, seed=None):
    minimal_imputation = []
    minimal_imputation_examples = []
    index_and_edge_repair = {}
    for index, example in enumerate(original_dataset):
        if np.isnan(example).any():
            repaired_dataset = generate_random_repair_with_edge(original_dataset, index_and_edge_repair, seed)
            is_support_vector, repair_for_this_example = checkImputationNecessity(repaired_dataset, original_dataset, labels, index, seed)
            index_and_edge_repair[index] = repair_for_this_example

            if is_support_vector:
                minimal_imputation.append([list(repair_for_this_example), index])
                minimal_imputation_examples.append(index)

    return minimal_imputation, minimal_imputation_examples, index_and_edge_repair

def get_Xy(data, label):
    if label== None:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return np.array(X), np.array(y)
    else:
        X = data.drop(label, axis=1)
        y = data[label]
    return np.array(X), np.array(y)

def sanity_check(X, minimal_imputation_examples):
    missing_rows = np.where(np.isnan(X).any(axis=1))[0]
    print("Missing rows:", missing_rows, "\nlength:", len(missing_rows))
    assert len(missing_rows) != len(minimal_imputation_examples), "Lengths do not match"
    examples_saved = [example for example in missing_rows if example not in minimal_imputation_examples]
    return len(examples_saved)


def mean_imputation(X_train, X_test, y_train, y_test, seed):
    start_time = time.time()
    mean_imputer = SimpleImputer(strategy='mean')
    X_train_mean_imputed = mean_imputer.fit_transform(X_train)
    model_mean = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed)
    model_mean.fit(X_train_mean_imputed, y_train)
    y_pred = model_mean.predict(X_test)
    accuracy_MI = model_mean.score(X_test, y_test)

    f1_MI = f1_score(y_test, y_pred)
    end_time = time.time()
    training_time_MI = end_time - start_time
    return accuracy_MI, f1_MI, training_time_MI

def knn_imputation(X_train, X_test, y_train, y_test, seed):
    start_time = time.time()
    knn_imputer = KNNImputer(n_neighbors=5)
    X_train_knn_imputed = knn_imputer.fit_transform(X_train)
    model_knn = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed)
    model_knn.fit(X_train_knn_imputed, y_train)
    y_pred = model_knn.predict(X_test)
    accuracy_KNN = model_knn.score(X_test, y_test)
    f1_KNN = f1_score(y_test, y_pred)
    end_time = time.time()
    training_time_KNN = end_time - start_time
    return accuracy_KNN, f1_KNN, training_time_KNN

def original_zero_imputation(OG_X_train, X_test, OG_y_train, y_test, seed):
    #combine train drop nulls from training set
    df_train = pd.concat([pd.DataFrame(OG_X_train), pd.Series(OG_y_train)], axis=1)
    df_train.dropna(inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    OG_X_train = df_train.iloc[:, :-1].values
    OG_y_train = df_train.iloc[:, -1].values


    start_time = time.time()
    model_OG = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed)
    model_OG.fit(OG_X_train, OG_y_train)
    y_pred = model_OG.predict(X_test)
    accuracy_OG = model_OG.score(X_test, y_test)
    f1_OG = f1_score(y_test, y_pred)
    end_time = time.time()
    training_time_OG = end_time - start_time
    return accuracy_OG, f1_OG, training_time_OG


import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


def generate_AC_data(df_train, df_test):
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    # Features and target split
    features, target = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    features_test, target_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

    # Identify rows with missing values
    ind = list(features[features.isna().any(axis=1)].index)
    not_ind = list(set(range(features.shape[0])) - set(ind))

    # Identify features with missing values
    feat = np.where(df_train.isnull().any())[0]

    # Impute missing values with small random values
    e_feat = np.copy(features)
    for i in ind:
        for j in feat:
            e_feat[i, j] = 0.01 * np.random.rand()

    return (
        features_test,
        target_test,
        csr_matrix(e_feat[not_ind, :]),
        np.ravel(target[not_ind]),
        csr_matrix(e_feat[ind, :]),
        np.ravel(target[ind]),
        csr_matrix(e_feat),
        np.arange(len(e_feat)).tolist(),
        ind,
        not_ind,
    )


def active_clean_driver(df_train, df_test):
    (
        features_test,
        target_test,
        X_clean,
        y_clean,
        X_dirty,
        y_dirty,
        X_full,
        train_indices,
        indices_dirty,
        indices_clean,
    ) = generate_AC_data(df_train, df_test)

    start_time = time.time()

    # Run ActiveClean process multiple times and average the results
    AC_records_1, AC_score_1 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_2, AC_score_2 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_3, AC_score_3 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_4, AC_score_4 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )
    AC_records_5, AC_score_5 = activeclean(
        (X_dirty, y_dirty),
        (X_clean, y_clean),
        (features_test, target_test),
        X_full,
        (train_indices, indices_dirty, indices_clean),
    )

    end_time = time.time()

    # Calculate the elapsed time and average score and records
    elapsed_time = end_time - start_time
    AC_time = elapsed_time / 5

    AC_records = (
                         AC_records_1 + AC_records_2 + AC_records_3 + AC_records_4 + AC_records_5
                 ) / 5
    AC_score = (AC_score_1 + AC_score_2 + AC_score_3 + AC_score_4 + AC_score_5) / 5

    return AC_records, AC_score, AC_time


if __name__ == '__main__':
    df = pd.read_csv("./Final-Datasets/water_potability.csv")
    label = 'Potability'
    X, y = get_Xy(df, label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    df_train = pd.concat([pd.DataFrame(X_train), pd.Series(y_train)], axis=1)
    df_test = pd.concat([pd.DataFrame(X_test), pd.Series(y_test)], axis=1)

    df_train.reset_index(drop=True, inplace=True)
    df_test.dropna(inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    total_examples = len(X_train)
    missing_values_per_row = pd.DataFrame(X_train).isnull().sum(axis=1)
    rows_with_missing_values = len(missing_values_per_row[missing_values_per_row > 0])

    print("Number of rows with missing values:", rows_with_missing_values)
    missing_factor = rows_with_missing_values / total_examples
    print(f"Total example {X_train.shape}, MISSING FACTOR : {missing_factor}")

    min_time = float('inf')
    best_seed = None
    best_minimal_imputation = None
    best_index_and_edge_repair = None
    best_number_of_examples_saved = None

    for attempt in range(1):  # trying different seeds
        print(f"################### Attempt {attempt + 1} ###################")
        seed = np.random.randint(0, 1000)
        start_time = time.time()
        minimal_imputation, imputation_required_examples_index, index_and_edge_repair = findminimalImputation(X_train.copy(), y_train.copy(), seed)
        end_time = time.time()
        attempt_time = end_time - start_time

        # handle for runs
        min_time = attempt_time
        best_seed = seed
        best_minimal_imputation = minimal_imputation
        best_index_and_edge_repair = index_and_edge_repair

    if best_minimal_imputation is not None:
        number_examples_saved = sanity_check(X_train, imputation_required_examples_index)
        print(f"Best seed: {best_seed} with minimal imputation set size: {len(best_minimal_imputation)} examples saved: {number_examples_saved}")
    else:
        print("No minimal imputation set found")

    accuracy_MI, f1_MI, training_time_MI = mean_imputation(X_train, X_test, y_train, y_test, seed)
    accuracy_KNN, f1_KNN, training_time_KNN = knn_imputation(X_train, X_test, y_train, y_test, seed)
    accuracy_OG, f1_OG, training_time_OG = original_zero_imputation(X_train, X_test, y_train, y_test,seed)

    examples_cleaned_AC, accuracy_AC ,training_time_AC = active_clean_driver(df_train, df_test)

    print(f"accuracy_MI: {accuracy_MI}, f1_MI: {f1_MI}, training_time_MI: {training_time_MI}")
    print(f"accuracy_KNN: {accuracy_KNN}, f1_KNN: {f1_KNN}, training_time_KNN: {training_time_KNN}")


    start_time = time.time()
    # Get the indices of examples that do not require imputation or are part of the minimal imputation set
    missing_rows = np.where(np.isnan(X_train).any(axis=1))[0]
    examples_saved = [example for example in missing_rows if example not in imputation_required_examples_index]

    # Remove examples saved from the training set
    valid_indices = [i for i in range(X_train.shape[0]) if i not in examples_saved]

    # Select the valid examples
    X_train_Minimal = X_train[valid_indices]
    y_train_Minimal = y_train[valid_indices]
    print(f"Length before minimal imputation: {len(X_train)}, Length after minimal imputation: {len(X_train_Minimal)}")

    # Mean imputation for remaining missing values
    X_train_Minimal_Mean = np.where(np.isnan(X_train_Minimal.copy()), np.nanmean(X_train_Minimal.copy(), axis=0), X_train_Minimal.copy())

    model = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=42, warm_start=True)
    model.fit(X_train_Minimal_Mean, y_train_Minimal)
    accuracy_MM = model.score(X_test, y_test)
    end_time = time.time()
    training_time_MM = end_time - start_time
    y_pred_MM = model.predict(X_test)
    f1_MM = f1_score(y_test, y_pred_MM)

    # KNN imputation for remaining missing values
    knn_imputer = KNNImputer(n_neighbors=5)
    X_train_Minimal = knn_imputer.fit_transform(X_train_Minimal)
    model = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=42, warm_start=True)
    model.fit(X_train_Minimal, y_train_Minimal)
    accuracy_MM_KNN = model.score(X_test, y_test)
    f1_MM_KNN = f1_score(y_test, model.predict(X_test))
    end_time = time.time()
    training_time_MM_KNN = end_time - start_time


    filename = os.path.join('./SVM/Final-Results', f"Water_MM.txt")
    with open(filename, "w+") as file:
        file.write(f"Number of Rows with missing values: {rows_with_missing_values}\n")
        file.write(f"Missing Factor: {missing_factor}\n")
        file.write(f"Checking Time (MM): {min_time}\n")
        file.write(f"Training Time (MM): {training_time_MM}\n")
        file.write(f"Examples Cleaned (MM): {len(best_minimal_imputation)}\n")
        file.write(f"Number of examples saved: {number_examples_saved}\n")
        file.write(f"Accuracy_MM_KNN: {accuracy_MM_KNN}, F1_MM_KNN: {f1_MM_KNN}, Training_Time_MM_KNN: {training_time_MM_KNN}\n")
        file.write(f"Accuracy_MM: {accuracy_MM}, F1_MM: {f1_MM}, Training_Time_MM: {training_time_MM}\n")
        file.write(f"Accuracy_MI: {accuracy_MI}, F1_MI: {f1_MI}, Training_Time_MI: {training_time_MI}\n")
        file.write(f"Accuracy_KNN: {accuracy_KNN}, F1_KNN: {f1_KNN}, Training_Time_KNN: {training_time_KNN}\n")
        file.write(f"Accuracy_OG: {accuracy_OG}, F1_OG: {f1_OG}, Training_Time_OG: {training_time_OG}\n")
        file.write(f"Accuracy_AC: {accuracy_AC}, Examples Cleanaed AC: {examples_cleaned_AC}, Training_Time_AC: {training_time_AC}\n")
        file.write(f"Best Seed: {best_seed}\n")
