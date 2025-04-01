import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import os
from sklearn.impute import SimpleImputer, KNNImputer
from hyperimpute.plugins.imputers import Imputers
# Generate random repairs and replace with edge repairs if available
def generate_random_repair_with_edge(dataset, index_and_edge_repair, random_seed=None):
    '''
    Generates a dataset with random repairs for missing values and replaces with edge repairs if available.

    Parameters:
        dataset (np.ndarray): The original dataset with missing values.
        index_and_edge_repair (dict): Dictionary containing indices and their corresponding edge repairs.
        random_seed (int, optional): Seed for random number generator. Default is None.

    Returns:
        np.ndarray: The dataset with random and edge repairs.
    '''
    if random_seed is not None:
        np.random.seed(random_seed)

    col_min = np.nanmin(dataset, axis=0)
    col_max = np.nanmax(dataset, axis=0)

    new_dataset = dataset.copy()
    # Replace with edge repairs if available
    for key, edge_repair in index_and_edge_repair.items():
        new_dataset[key] = edge_repair

    # Create a mask of where the NaNs are
    nan_mask = np.isnan(new_dataset)

    # Generate random repairs for NaNs
    random_repaired_dataset = new_dataset.copy()
    for i in range(new_dataset.shape[1]):  # Iterate over each column
        random_repaired_dataset[nan_mask[:, i], i] = np.random.uniform(col_min[i], col_max[i], size=nan_mask[:, i].sum())

    return random_repaired_dataset


# Function to find edge repair for a missing value
def findEdgeRepair(incomplete_example, repaired_other_examples, model, y_incompleteExample):
    '''
    Finds the edge repair for a given incomplete example.

    Parameters:
        incomplete_example (np.ndarray): The incomplete example with missing values.
        repaired_other_examples (np.ndarray): The dataset with repaired examples.
        model (SGDClassifier): The trained model.
        y_incompleteExample (int): The label of the incomplete example.

    Returns:
        np.ndarray: The repaired example.
    '''
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


# Check if an incomplete example can be a support vector
def checkSV(incompleteExample, repairedOtherExamples, labels, y_incompleteExample, seed=None):
    '''
    Checks if an incomplete example can be a support vector after repairs.

    Parameters:
        incompleteExample (np.ndarray): The incomplete example with missing values.
        repairedOtherExamples (np.ndarray): The dataset with repaired examples.
        labels (np.ndarray): The labels for the dataset.
        y_incompleteExample (int): The label of the incomplete example.

    Returns:
        tuple: (bool, np.ndarray) indicating if it is a support vector and the repaired example.
    '''
    model = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=42, warm_start=True)
    model.fit(repairedOtherExamples, labels)

    repaired_example = findEdgeRepair(incompleteExample, repairedOtherExamples, model, y_incompleteExample)

    decision_value = np.dot(model.coef_, repaired_example) + model.intercept_
    product = y_incompleteExample * decision_value

    if product < 1:
        return True, repaired_example

    return False, repaired_example


# Check the necessity of imputation for an incomplete example
def checkImputationNecessity(possible_repaired_dataset, original_dataset, labels, example_index, seed=None):
    '''
    Checks if imputation is necessary for an incomplete example.

    Parameters:
        possible_repaired_dataset (np.ndarray): The dataset with possible repairs.
        original_dataset (np.ndarray): The original dataset with missing values.
        labels (np.ndarray): The labels for the dataset.
        example_index (int): The index of the incomplete example.

    Returns:
        tuple: (bool, np.ndarray) indicating if imputation is necessary and the repaired example.
    '''
    possible_repaired_data_subset = np.delete(possible_repaired_dataset, example_index, axis=0)
    labels_subset = np.delete(labels, example_index, axis=0)
    if possible_repaired_data_subset.shape[0] == 0 or len(np.unique(labels_subset)) < 2:
        raise ValueError("Insufficient data or classes")

    incomplete_example = original_dataset[example_index]
    return checkSV(incomplete_example, possible_repaired_data_subset, labels_subset, labels[example_index], seed)


# Generate minimal imputation set
def findminimalImputation(original_dataset, labels, seed=None):
    '''
    Generates the minimal imputation set for the dataset.

    Parameters:
        original_dataset (np.ndarray): The original dataset with missing values.
        labels (np.ndarray): The labels for the dataset.

    Returns:
        tuple: (list, list, dict) indicating the minimal imputation set, the indices of examples requiring imputation,
               and the index and edge repair dictionary.
    '''
    minimal_imputation = []
    minimal_imputation_examples = []
    index_and_edge_repair = {}
    for index, example in enumerate(original_dataset):
        if np.isnan(example).any():  # for each missing example in the dataset
            repaired_dataset = generate_random_repair_with_edge(original_dataset, index_and_edge_repair, seed)
            is_support_vector, repair_for_this_example = checkImputationNecessity(repaired_dataset, original_dataset, labels, index, seed)
            index_and_edge_repair[index] = repair_for_this_example

            if is_support_vector:
                minimal_imputation.append([list(repair_for_this_example), index])
                minimal_imputation_examples.append(index)
                pass  # Once we find one imputation necessity, we don't need to check further combinations for this example

    return minimal_imputation, minimal_imputation_examples, index_and_edge_repair


def get_Xy(data, label):
    '''
    Splits the dataset into features and labels.

    Parameters:
        data (pd.DataFrame): The dataset.
        label (str): The name of the label column.

    Returns:
        tuple: (np.ndarray, np.ndarray) containing features and labels.
    '''
    X = data.drop(label, axis=1)
    y = data[label]
    return np.array(X), np.array(y)

def sanity_check(X, minimal_imputation_examples):
    '''
    Performs a sanity check to ensure the minimal imputation examples are correct.

    Parameters:
        X (np.ndarray): The dataset with features.
        minimal_imputation_examples (list): List of indices of examples requiring imputation.

    Returns:
        None: Prints the missing rows and their length.
    '''
    missing_rows = np.where(np.isnan(X).any(axis=1))[0]
    print("Missing rows:", missing_rows, "\nlength:", len(missing_rows))
    assert len(missing_rows) != len(minimal_imputation_examples), "Lengths do not match"
    #return examples dont need imputation
    examples_saved = [example for example in missing_rows if example not in minimal_imputation_examples]
    return len(examples_saved)
def mean_imputation(X_train, X_test, y_train, y_test):
    '''
    Imputes missing values with the mean of the column.

    Parameters:
        X (np.ndarray): The dataset with features.
        y (np.ndarray): The labels for the dataset.


    Returns:
        float: The accuracy of the model.
        float: The training time of the model.
    '''
    # Mean Imputation
    start_time = time.time()
    mean_imputer = SimpleImputer(strategy='mean')
    X_train_mean_imputed = mean_imputer.fit_transform(X_train)
    model_mean = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=42, warm_start=True)
    model_mean.fit(X_train_mean_imputed, y_train)
    accuracy_MI = model_mean.score(X_test, y_test)
    end_time = time.time()
    training_time_MI = end_time - start_time
    return accuracy_MI, training_time_MI


def miwae_imputation(X_train, X_test, y_train, y_test):
    start_time_miwae = time.time()
    method='miwae'
    # Simple imputation using mean strategy for each column
    plugin = Imputers().get(method)
    imputed_X = plugin.fit_transform(X_train)
    # Assert that there are no more null values in X_train
    assert not imputed_X.isnull().any().any()

    clf = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=42, warm_start=True)
    clf.fit(imputed_X, y_train)
    score = clf.score(X_test, y_test)
    end_time_miwae = time.time()
    miwae_time = end_time_miwae - start_time_miwae
    return score, miwae_time
def knn_imputation(X_train, X_test, y_train, y_test):
    start_time = time.time()
    knn_imputer = KNNImputer(n_neighbors=5)
    X_train_knn_imputed = knn_imputer.fit_transform(X_train)
    model_knn = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=42, warm_start=True)
    model_knn.fit(X_train_knn_imputed, y_train)
    accuracy_KNN = model_knn.score(X_test, y_test)
    end_time = time.time()
    training_time_KNN = end_time - start_time
    return accuracy_KNN, training_time_KNN

if __name__ == '__main__':
    df = pd.read_csv("../Final-Datasets/Online-Education.csv")
    label = 'Preferred device for an online course_Mobile'
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
        # seed = np.random.randint(0, 1000)
        seed=42
        start_time = time.time()
        minimal_imputation, imputation_required_examples_index, index_and_edge_repair = findminimalImputation(X_train.copy(), y_train.copy(), seed)
        end_time = time.time()
        attempt_time = end_time - start_time

        if len(minimal_imputation) > 0 and attempt_time < min_time:
            min_time = attempt_time
            best_seed = seed
            best_minimal_imputation = minimal_imputation
            best_index_and_edge_repair = index_and_edge_repair

    if best_minimal_imputation is not None:
        number_examples_saved = sanity_check(X_train, imputation_required_examples_index)
        print(f"Best seed: {best_seed} with minimal imputation set size: {len(best_minimal_imputation)} examples saved: {number_examples_saved}")
    else:
        print("No minimal imputation set found")

    accuracy_MI, time_MI = mean_imputation(X_train.copy(), X_test, y_train.copy(), y_test)
    accuracy_KNN, time_KNN = knn_imputation(X_train.copy(), X_test, y_train.copy(), y_test)
    accuracy_Miwae, time_Miwae = miwae_imputation(X_train.copy(), X_test, y_train.copy(), y_test)

    X_train_Minimal = X_train.copy()
    y_train_Minimal = y_train.copy()

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
    X_train_Minimal = np.where(np.isnan(X_train_Minimal), np.nanmean(X_train_Minimal, axis=0), X_train_Minimal)


    model = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=42, warm_start=True)
    model.fit(X_train_Minimal, y_train_Minimal)
    accuracy_MM = model.score(X_test, y_test)
    end_time = time.time()
    training_time_MM = end_time - start_time

    print(f"Accuracy (MM): {accuracy_MM}, Training & Checking Time (MM): {training_time_MM + min_time}, Accuracy (MI): {accuracy_MI}, Training Time (MI): {time_MI}, Accuracy (KNN): {accuracy_KNN}, Training Time (KNN): {time_KNN}")

    filename = os.path.join('../Final-Results', f"Online_Education_MM.txt")
    with open(filename, "w+") as file:
        file.write(f"Number of Rows with missing values: {rows_with_missing_values}\n")
        file.write(f"Missing Factor: {missing_factor}\n")
        file.write(f"Checking Time (MM): {min_time}\n")
        file.write(f"Training Time (MM): {training_time_MM}\n")
        file.write(f"Examples Cleaned (MM): {len(best_minimal_imputation)}\n")
        file.write(f"Number of examples saved: {number_examples_saved}\n")
        file.write(f"Accuracy (MM): {accuracy_MM}\n")
        file.write(f"Accuracy (MI): {accuracy_MI}\n")
        file.write(f"Training Time (MI): {time_MI}\n")
        file.write(f"Accuracy (KNN): {accuracy_KNN}\n")
        file.write(f"Training Time (KNN): {time_KNN}\n")
        file.write(f"Accuracy (Miwae): {accuracy_Miwae}\n")
        file.write(f"Training Time (Miwae): {time_Miwae}\n")
        file.write(f"Best Seed: {best_seed}\n")
