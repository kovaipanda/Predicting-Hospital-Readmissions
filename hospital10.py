import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Read the CSV file
file_path_cleaned = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data6.csv'
df = pd.read_csv(file_path_cleaned)

# Identify the target column
target_column = "Readmitted"

# Specify the columns to use for interpolation
columns_for_interpolation = [
    "Age", "Num_Lab_Procedures", "Num_Medications",  "Num_Diagnoses",'Num_Outpatient_Visits','Num_Emergency_Visits','Num_Inpatient_Visits','Patient_ID'
]
#"Num_Outpatient_Visits","Num_Inpatient_Visits", "Num_Emergency_Visits"

# Separate features and target
X = df[columns_for_interpolation]
y = df[target_column]

# Check for missing values
print("Missing values before interpolation:")
print(X.isnull().sum())

# Interpolate missing values using IterativeImputer (MICE)
imputer = IterativeImputer(random_state=42)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Check for missing values after interpolation
print("\nMissing values after interpolation:")
print(X_imputed.isnull().sum())

# Combine imputed features with the target column
df_imputed = pd.concat([X_imputed, y], axis=1)

# Save the preprocessed data to a new CSV file
df_imputed.to_csv("synthetic_hospital_readmissions_data7.csv", index=False)


# Display the first few rows of the preprocessed data
print("\nFirst few rows of the preprocessed data:")
print(df_imputed.head())