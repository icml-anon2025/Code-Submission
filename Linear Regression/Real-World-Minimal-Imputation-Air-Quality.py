import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from hyperimpute.plugins.imputers import Imputers
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

def visualize_residuals_distribution(r):
    # Plot the distribution of the residuals using a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(r, kde=True, bins=30, color='blue', stat='density')
    plt.title("Distribution of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.show()

# Function to perform Orthogonal Matching Pursuit and return must-impute features
def omp_select_features(X, y, threshold, max_iter=100):
    numNeedingImputation = 0
    #r = y.copy()

    # Impute the missing values in the incomplete features with the mean of that feature
    X_impute = X.copy()
    X_impute.fillna(X_impute.mean(), inplace=True)

    # Ensure there are no NaN values left after imputation
    assert not X_impute.isna().any().any(), "There are still NaN values in X_impute."
    assert not y.isna().any(), "There are NaN values in the target vector y."

    # Initialize the set of selected features indices (includes all complete features)
    complete_features = X.columns[X.notna().all()].tolist()
    S = [X.columns.get_loc(feature) for feature in complete_features]  # indices of complete features
    print("Number of complete features:", len(S))

    # Identify incomplete features (any with missing values)
    incomplete_features = X.columns[X.isna().any()].tolist()
    incomplete_features_indices = [X.columns.get_loc(feature) for feature in incomplete_features]
    print("Number of incomplete features:", len(incomplete_features_indices))

    # Initialize the residual vector as the label vector y projected onto the column space of the complete features
    if complete_features:
        # Fit a linear regression model to the complete features
        model = LinearRegression()
        model.fit(X_impute[complete_features], y)

        # Compute the predicted values (projection of y)
        y_pred = model.predict(X_impute[complete_features])

        # Compute the residual vector
        r = y - y_pred


    remaining_features = incomplete_features_indices
    print(f"Features with missing values: {remaining_features}")

    for _ in range(max_iter):
        if not remaining_features:
            break
        
        # Calculate the dot product of each remaining feature vector with the residual vector
        dot_products = X_impute.iloc[:, remaining_features].T @ r

        norms = np.linalg.norm(X_impute.iloc[:, remaining_features], axis=0) * np.linalg.norm(r)

        # Add a small epsilon to the norms to avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)

        cosine_similarities = np.abs(dot_products / norms)
        
        # Check for NaN entries in dot_products and cosine_similarities
        nan_dot_products_count = np.sum(np.isnan(dot_products))
        nan_cosine_similarities_count = np.sum(np.isnan(cosine_similarities))


        # Find the maximum cosine similarity
        max_cosine_similarity = np.max(cosine_similarities)

        if threshold > 0 and max_cosine_similarity < threshold:
            print(f"Max Cosine Similarity is below threshold, stopping.")
            break

        # Find the feature vector that has the maximum normalized dot product
        j = remaining_features[np.argmax(cosine_similarities)]
        S.append(j)
        remaining_features.remove(j)
        numNeedingImputation += 1


        # Stop early if the maximum normalized dot product is nan
        if np.isnan(max_cosine_similarity):
            print("Max Cosine Similarity is NaN, stopping.")
            break
        
        # Fit a linear regression model to the selected features
        model = LinearRegression()
        model.fit(X_impute.iloc[:, S], y)

        # Compute the predicted values
        y_pred = model.predict(X_impute.iloc[:, S])

        # Update the residual vector
        r = y - y_pred

    return (S, numNeedingImputation)

# Function to evaluate the model with different imputation strategies
def evaluate_model(X_train, X_test, y_train, y_test, impute_strategy, must_impute=None):
    if impute_strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif impute_strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif impute_strategy == 'min':
        imputer = None  # No imputer needed for min strategy

    # Impute only the selected must-impute features
    X_train_impute = X_train.copy()
    X_test_impute = X_test.copy()

    start_time = time.time()  # Start timing

    # If we are imputing everything (baseline)
    if must_impute is None:
        print("Imputing all features...")
        if impute_strategy == 'min':
            for feature in range(X_train.shape[1]):
                min_value = X_train.iloc[:, feature].min()
                X_train_impute.iloc[:, feature].fillna(min_value, inplace=True)
                X_test_impute.iloc[:, feature].fillna(min_value, inplace=True)
        else:
            X_train_impute = imputer.fit_transform(X_train)
            X_test_impute = imputer.transform(X_test)
    
    elif len(must_impute) == 0:
        print("No features were selected to be imputed, removing columns with missing values.")
        # Get the indices of columns with missing values
        missing_value_columns = X_train.columns[X_train.isna().any()].tolist()

        # Remove those columns from the training and testing sets
        X_train_impute = X_train_impute.drop(columns=missing_value_columns)
        X_test_impute = X_test_impute.drop(columns=missing_value_columns)

    else:
        if impute_strategy == 'min':
            for feature in must_impute:
                min_value = X_train.iloc[:, feature].min()
                X_train_impute.iloc[:, feature].fillna(min_value, inplace=True)
                X_test_impute.iloc[:, feature].fillna(min_value, inplace=True)
        else:
            X_train_impute.iloc[:, must_impute] = imputer.fit_transform(X_train.iloc[:, must_impute])
            X_test_impute.iloc[:, must_impute] = imputer.transform(X_test.iloc[:, must_impute])

        # Get the indices of columns with missing values
        missing_value_columns = X_train_impute.columns[X_train_impute.isna().any()].tolist()

        # Remove those columns from the training and testing sets
        X_train_impute = X_train_impute.drop(columns=missing_value_columns)
        X_test_impute = X_test_impute.drop(columns=missing_value_columns)

    X_train_impute = np.asarray(X_train_impute)
    X_test_impute = np.asarray(X_test_impute)
    assert not np.isnan(X_train_impute).any(), "There are still NaN values in X_train_impute."
    assert not np.isnan(X_test_impute).any(), "There are still NaN values in X_test_impute."

    elapsed_time = time.time() - start_time  # End timing

    model = LinearRegression()
    model.fit(X_train_impute, y_train)


    coef = model.coef_

    # Recreate the predictions including the removed features with zero coefficients
    y_pred_full = X_test_impute @ coef

    # Calculate the mean squared error with the full predictions (0 coefficient for removed features)
    mse = mean_squared_error(y_test, y_pred_full)
    
    return mse, elapsed_time

# Load data
print("Loading data...")

data = pd.read_csv('./Linear Regression/datasets/cleaned_air_quality.csv')

# Split data into features and target
X = data.drop(columns=['T'])
y = data['T']


# Split into training and testing sets (for cross-validation)
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline performance by imputing all missing values
print("Calculating baseline performance...")
mse_mean_baseline, time_mean_baseline = evaluate_model(X_train, X_test, y_train, y_test, 'mean')
mse_knn_baseline, time_knn_baseline = evaluate_model(X_train, X_test, y_train, y_test, 'knn')
mse_min_baseline, time_min_baseline = evaluate_model(X_train, X_test, y_train, y_test, 'min')

print(f"Baseline MSE (MI): {mse_mean_baseline:.4f}, Time: {time_mean_baseline:.4f} seconds")
print(f"Baseline MSE (KI): {mse_knn_baseline:.4f}, Time: {time_knn_baseline:.4f} seconds")
print(f"Baseline MSE (MinI): {mse_min_baseline:.4f}, Time: {time_min_baseline:.4f} seconds")

# Gradually tune down the threshold and evaluate performance
print("Starting threshold tuning...")
# thresholds = np.linspace(1, 0, 20)  # Adjust range as needed
thresholds = np.logspace(-1, -7, num=10)
# thresholds = [1e-16]
results = []
print("\n")
 
for threshold in thresholds:
    #print(f"Threshold: {threshold:.2f}")
    print("Threshold:", threshold)
    # To find the must-impute features, impute the missing datapoints in features with the mean value
    must_impute_features, numNeedingImputation = omp_select_features(X, y, threshold)

    print(f"Number of Must-Impute Features: {numNeedingImputation}")
    mse_mean, time_mean = evaluate_model(X_train, X_test, y_train, y_test, 'mean', must_impute=must_impute_features)
    mse_knn, time_knn = evaluate_model(X_train, X_test, y_train, y_test, 'knn', must_impute=must_impute_features)
    mse_min, time_min = evaluate_model(X_train, X_test, y_train, y_test, 'min', must_impute=must_impute_features)
    
    print(f"MSE (MI): {mse_mean:.4f}, Time: {time_mean:.4f} seconds")
    print(f"MSE (KI): {mse_knn:.4f}, Time: {time_knn:.4f} seconds")
    print(f"MSE (MinI): {mse_min:.4f}, Time: {time_min:.4f} seconds")
    print("")

    
    results.append({
        'Threshold': f"{threshold:.4f}",
        '# Features Imputed': numNeedingImputation,
        'MSE (MI)': f"{mse_mean:.4f}",
        'Time (MI)': f"{time_mean:.4f}",
        'MSE (KI)': f"{mse_knn:.4f}",
        'Time (KI)': f"{time_knn:.4f}",
        'MSE (MinI)': f"{mse_min:.4f}",
        'Time (MinI)': f"{time_min:.4f}"
    })

# print results to csv
results_df = pd.DataFrame(results)
results_df.to_csv('./results/airQualityResults.csv', index=False)


print("Results saved to results.csv")
print(f"Baseline MSE (MI): {mse_mean_baseline:.4f}, Time: {time_mean_baseline:.4f} seconds")
print(f"Baseline MSE (KI): {mse_knn_baseline:.4f}, Time: {time_knn_baseline:.4f} seconds")
print(f"Baseline MSE (MinI): {mse_min_baseline:.4f}, Time: {time_min_baseline:.4f} seconds")

# print baseline results to a seperate csv
baseline_results = {
    '# Features Imputed': len(X.columns[X.isna().any()].tolist()),
    'Baseline MSE (MI)': f"{mse_mean_baseline:.4f}",
    'Time (MI)': f"{time_mean_baseline:.4f}",
    'Baseline MSE (KI)': f"{mse_knn_baseline:.4f}",
    'Time (KI)': f"{time_knn_baseline:.4f}",
    'Baseline MSE (MinI)': f"{mse_min_baseline:.4f}",
    'Time (MinI)': f"{time_min_baseline:.4f}"
}

baseline_results_df = pd.DataFrame([baseline_results])
baseline_results_df.to_csv('./results/airQualityResultsBaseline.csv', index=False)

