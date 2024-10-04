import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data2.csv'
data = pd.read_csv(file_path)

# Select numeric columns for normalization
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the selected numeric columns
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Save the normalized dataset to a new CSV file
normalized_file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data3.csv'
data.to_csv(normalized_file_path, index=False)

print(f"Normalized data saved to: {normalized_file_path}")
