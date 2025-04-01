from sympy import symbols, Matrix
import sympy as sp
import pandas as pd
from tqdm import tqdm

# Function to project vector a onto vector b
def project(a, b):
    return (a.dot(b) / b.dot(b)) * b

# Function to calculate the residual vector
def calculate_residual(label_vector, feature_vectors):
    residual = label_vector
    for col in tqdm(range(feature_vectors.shape[1]), desc="Calculating residual"):
        feature = feature_vectors.col(col) 
        residual -= project(residual, feature)

    return residual

# Function to Compute Row Echelon Form (REF) with symbolic computation
def get_ref_and_pivots(matrix):
    ref_matrix, pivot_columns = matrix.T.rref()
    return ref_matrix.T, pivot_columns

# Function to simplify symbolic matrix
def simplify_matrix(matrix):
    print("Simplifying matrix...")
    return matrix.applyfunc(sp.simplify)

# Load the dataset
file_path = 'datasets/cleaned_air_quality.csv'
df = pd.read_csv(file_path)
df = df.head(20)

print("Replacing missing values with unique symbols...")
# Step 1: Replace missing values with unique symbols
for col in df.columns:
    if df[col].isnull().any():
        symbol = symbols(f'missing_{col}')
        df[col].fillna(symbol, inplace=True)
        
print("DataFrame:")
print(df)

# # Extract label vector and feature matrix
label_vector = Matrix(df['T'].values)
feature_vectors = Matrix(df.drop(columns=['T']).values.tolist())

# Step 2: Compute Row Echelon Form (REF) and get pivot columns
# print("Computing linearly independant features...")
# ref_matrix, pivot_columns = get_ref_and_pivots(feature_vectors)
# independent_features = feature_vectors[:, pivot_columns]

# If not all vectors are linearly independent, exit the program (cannot guarantee minimality)
# Questions: Can we guarantee if all complete features are LI?
# https://github.com/ResidentMario/missingno

# Step 3: Simplify feature vectors
feature_vectors = simplify_matrix(feature_vectors)

print("Simplified feature vectors:")
for row in feature_vectors.tolist():
    print(row)

# Step 4: Calculate the residual vector using the independent features
print("Calculating residual...")
residual_vector = calculate_residual(label_vector, feature_vectors)
residual_vector = simplify_matrix(residual_vector)
print("Simplified Residual vector:", residual_vector.evalf(5), end="\n\n")

# # Step 5: Define vector b
b = label_vector - residual_vector

# # Step 6: Solve the system A * x = b
A = feature_vectors
print("Solving the system A * x = b...")

try:
    # Attempt to solve using least squares 
    # solution = A.solve_least_squares(b, method='CH')    # CH    = Cholesky decomposition
    solution = A.solve_least_squares(b, method='LDL')     # LDL   = LDLsolve
    # solution = A.solve_least_squares(b, method='QR')    # QR    = QRsolve decomposition
    # solution = A.solve_least_squares(b, method='PINV')  # PINV  = pinv_solve decomposition
    # solution = A.solve_least_squares(b)                 # Default

except Exception as e:
    print("Exception", e)
    print("Matrix is non-invertible, using gauss_jordan_solve instead...")
    solution, _ = A.gauss_jordan_solve(b)
    solution = simplify_matrix(solution)

# # Step 7: Pretty print the solution
print("Formatting solution...")
feature_names = df.drop(columns=['T']).columns

# Dictionary of feature : coefficient pairs
solution_dict = {name: coef for name, coef in zip(feature_names, solution)}

# # Step 8: Convert the solution to a DataFrame for better visualization
solution_df = pd.DataFrame(list(solution_dict.items()), columns=['Feature', 'Coefficient'])
print("\nCoefficients DataFrame:")
print(solution_df)


 