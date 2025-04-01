import pandas as pd

def calculate_missing_factor(csv_file):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Calculate the total number of examples
    total_examples = len(df)

    # Calculate the number of incomplete examples (rows with at least one missing value)
    incomplete_examples = df.isnull().any(axis=1).sum()

    # Calculate the missing factor (MF)
    missing_factor = incomplete_examples / total_examples

    return missing_factor

# Path to your CSV file
csv_file = 'datasets/cleaned_air_quality.csv'

# Calculate and print the missing factor
mf = calculate_missing_factor(csv_file)
print(f"Missing Factor (MF): {mf:.4f}")