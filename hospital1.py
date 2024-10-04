import pandas as pd

# Corrected file path (remove extra quotes)
file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data.csv'

# Load the file into a DataFrame
df = pd.read_csv(file_path)

# Count the total number of missing values in the entire DataFrame
total_missing_cells = df.isnull().sum().sum()

# Count the number of missing values per column
missing_values_per_column = df.isnull().sum()

# Display the total number of missing cells
print(f"Total number of missing cells: {total_missing_cells}\n")

# Display the column names and their respective missing values
print("Column Name and Number of Missing Values:")
print(missing_values_per_column)  # Show all columns, including those without missing values
