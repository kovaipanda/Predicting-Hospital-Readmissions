import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path_cleaned = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data7.csv'
df = pd.read_csv(file_path_cleaned)

# Define the target column
target_col = 'Readmitted'

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols.remove(target_col)  # Remove target column from numerical columns


# Visualization for Numerical Features vs Target
plt.figure(figsize=(16, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, len(numerical_cols)//2 + 1, i)
    sns.boxplot(x=target_col, y=col, data=df)
    plt.title(f'{col} vs {target_col}')
plt.tight_layout()
plt.show()


