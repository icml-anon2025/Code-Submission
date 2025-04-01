import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

# Load the dataset
df = pd.read_csv('datasets/Fatal-Shotting.csv')

print(df.head())

# Initialize the StandardScaler
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

# Normalize the entire dataset
normalized_data = scaler.fit_transform(df)

# Create a DataFrame for the normalized dataset
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

print(normalized_df.head())

# Save the normalized dataset to a new CSV file
normalized_df.to_csv('fatal_shooting_normalized.csv', index=False)

print("Normalization complete. The dataset has been saved as 'fatal_shooting_normalized.csv'.")