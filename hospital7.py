import pandas as pd

# Load the original dataset
file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data4.csv'
df = pd.read_csv(file_path)

# Define the columns to discard
columns_to_discard = [ 'Gender', 'Admission_Type', 'Diagnosis', 'A1C_Result']

# Drop the specified columns
df_cleaned = df.drop(columns=columns_to_discard)

# Save the cleaned DataFrame to a new CSV file
output_file_path = r'C:\Users\DELL\Downloads\hospital_resubmisssion\synthetic_hospital_readmissions_data5.csv'
df_cleaned.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to: {output_file_path}")

