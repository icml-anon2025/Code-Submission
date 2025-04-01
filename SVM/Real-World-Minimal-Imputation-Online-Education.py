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

if __name__ == '__main__':
    df = pd.read_csv("./Final-Datasets/Online-Education.csv")
    label = 'Preferred device for an online course_Mobile'

    # Store the original indices from the original dataframe
    original_indices = df.index

    # Get features and target variable
    X, y = get_Xy(df, label)

    # Perform the train-test split
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
    accuracy_OG, f1_OG, training_time_OG = original_zero_imputation(X_train, X_test, y_train, y_test, seed)

    print(f"accuracy_MI: {accuracy_MI}, f1_MI: {f1_MI}, training_time_MI: {training_time_MI}")
    print(f"accuracy_KNN: {accuracy_KNN}, f1_KNN: {f1_KNN}, training_time_KNN: {training_time_KNN}")


    start_time = time.time()
    # Get the indices of examples that do not require imputation or are part of the minimal imputation set
    missing_rows = np.where(np.isnan(X_train).any(axis=1))[0]
    examples_saved = [example for example in missing_rows if example not in imputation_required_examples_index]

    #print indices of examples saved

    # Remove examples saved from the training set
    valid_indices = [i for i in range(X_train.shape[0]) if i not in examples_saved]

    # Select the valid examples i.e examples that do not require imputation or are not part of the minimal imputation set
    X_train_Minimal = X_train[valid_indices]
    y_train_Minimal = y_train[valid_indices]
    print(f"Length before minimal imputation: {len(X_train)}, Length after minimal imputation: {len(X_train_Minimal)}")

    # Map back the original indices using valid_indices
    # `original_indices[valid_indices]` will give us the original row indices corresponding to the valid rows
    imputedf = pd.concat(
        [pd.DataFrame(X_train_Minimal, index=original_indices[valid_indices]),
         pd.Series(y_train_Minimal, index=original_indices[valid_indices])],
        axis=1
    )

    # Save the result with original indices
    filename = os.path.join('./SVM/Final-Results', f"Online_Education_MinimalX.csv")
    imputedf.to_csv(filename, index=True)
    print(f"Saved minimal imputation data to {filename}")
    print(imputedf.head(5))
    # egt number of missing values in imputedf
    missing_values_per_row = pd.DataFrame(imputedf).isnull().sum(axis=1)
    rows_with_missing_values = len(missing_values_per_row[missing_values_per_row > 0])
    print(f"Number of rows with missing values in imputedf: {rows_with_missing_values}")
    ####### we will use this for MinMI , MinKNN and MinDiffI

    # Mean imputation for remaining missing values
    X_train_Minimal_Mean = np.where(np.isnan(X_train_Minimal.copy()), np.nanmean(X_train_Minimal.copy(), axis=0),
                                    X_train_Minimal.copy())

    model = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=42, warm_start=True)
    model.fit(X_train_Minimal_Mean, y_train_Minimal)
    accuracy_MM = model.score(X_test, y_test)
    end_time = time.time()
    training_time_MM = end_time - start_time
    y_pred_MM = model.predict(X_test)
    f1_MM = f1_score(y_test, y_pred_MM)

    #print number of missing values in X_train_Minimal
    missing_values_per_row = pd.DataFrame(X_train_Minimal).isnull().sum(axis=1)
    rows_with_missing_values = len(missing_values_per_row[missing_values_per_row > 0])
    print(f"Number of rows with missing values in X_train_Minimal: {rows_with_missing_values}")
    # KNN imputation for remaining missing values
    knn_imputer = KNNImputer(n_neighbors=5)
    X_train_Minimal = knn_imputer.fit_transform(X_train_Minimal)
    model = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=42, warm_start=True)
    model.fit(X_train_Minimal, y_train_Minimal)
    accuracy_MM_KNN = model.score(X_test, y_test)
    f1_MM_KNN = f1_score(y_test, model.predict(X_test))
    end_time = time.time()
    training_time_MM_KNN = end_time - start_time


    filename = os.path.join('./SVM/Final-Results', f"Online_Education_MM.txt")
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
        file.write(f"Best Seed: {best_seed}\n")
