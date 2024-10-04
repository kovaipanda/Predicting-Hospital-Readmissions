import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the cleaned dataset
file_path_cleaned = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data3.csv'
df_cleaned = pd.read_csv(file_path_cleaned)

# Specify the target column and features
target_column = 'Readmitted'  # replace with your actual target column name
X = df_cleaned.drop(columns=[target_column])  # Features
y = df_cleaned[target_column]  # Target variable

# Print class distribution before balancing
print("Class distribution before balancing:", Counter(y))

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine resampled features and target back into a DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled[target_column] = y_resampled

# Print class distribution after balancing
print("Class distribution after balancing:", Counter(y_resampled))

# Save the new balanced dataset to a CSV file
balanced_file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data4.csv'
df_resampled.to_csv(balanced_file_path, index=False)

print(f"Balanced dataset saved to: {balanced_file_path}")
