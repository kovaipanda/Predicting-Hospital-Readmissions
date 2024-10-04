import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path_cleaned = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data5.csv'
df_cleaned = pd.read_csv(file_path_cleaned)

# Define a function to remove outliers using IQR
def remove_outliers_iqr(df):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create a mask for non-outliers
    mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
    
    # Return the DataFrame without outliers
    return df[mask]

# Visualize box plots before removing outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_cleaned[['Patient_ID','Age', 'Num_Lab_Procedures', 'Num_Medications', 'Num_Diagnoses','Num_Outpatient_Visits','Num_Emergency_Visits','Num_Inpatient_Visits']])
plt.title("Box Plots Before Removing Outliers")
plt.show()

# Apply the function to remove outliers
df_no_outliers = remove_outliers_iqr(df_cleaned)

# Visualize box plots after removing outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_no_outliers[['Patient_ID','Age', 'Num_Lab_Procedures', 'Num_Medications', 'Num_Diagnoses','Num_Outpatient_Visits','Num_Emergency_Visits','Num_Inpatient_Visits']])
plt.title("Box Plots After Removing Outliers")
plt.show()

# Save the cleaned dataset without outliers
file_path_no_outliers = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data6.csv'
df_no_outliers.to_csv(file_path_no_outliers, index=False)

print(f"Data without outliers saved to: {file_path_no_outliers}")
