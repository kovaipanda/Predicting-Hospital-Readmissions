# Predicting-Hospital-Readmissions

Project Overview

This project involves analyzing synthetic hospital readmissions data to predict whether a patient will be readmitted. 

The data includes patient demographics, medical procedures, and various categorical and numerical features.

The primary goal is to preprocess the data, handle missing values, perform exploratory data analysis, and build machine learning models for prediction.

Project Structure

The project is structured into multiple Python scripts that each perform specific tasks related to data preprocessing, exploration, model training, and evaluation.

Files and Scripts:

hospital1.py:

Task:
Handle missing values by counting them in each column.

Description:
This script loads the synthetic hospital readmissions dataset and calculates the total number of missing cells as well as missing values per column. The results are printed to the console.

Column Name and Number of Missing Values:
Patient_ID                  0
Age                         0
Gender                      0
Admission_Type              0
Diagnosis                   0
Num_Lab_Procedures          0
Num_Medications             0
Num_Outpatient_Visits       0
Num_Inpatient_Visits        0
Num_Emergency_Visits        0
Num_Diagnoses               0
A1C_Result               4034
Readmitted                  0
dtype: int64

hospital2.py:

Task:
Perform chi-square tests for categorical features.

Description:
This script evaluates the relationships between categorical features and the target variable (Readmitted).
It uses chi-square tests to determine the strength of these relationships and stores the results in a CSV file.

Chi-square Test Results:
          Feature  Chi2 Statistic   P-Value  Degrees of Freedom
0          Gender        0.737870  0.691470                   2
1  Admission_Type        2.140936  0.342848                   2
2       Diagnosis        3.425236  0.330589                   3
3      A1C_Result        2.173099  0.140443                   1

hospital3.py:

Task:
Label encoding for categorical variables.

Description:
This script encodes categorical variables in the dataset into numerical values using LabelEncoder.
It also saves the encoded dataset and prints the mappings of categorical columns to their respective encodings.

Mappings of Categorical Columns:
Gender: {'Female': 0, 'Male': 1, 'Other': 2}
Admission_Type: {'Elective': 0, 'Emergency': 1, 'Urgent': 2}
Diagnosis: {'Diabetes': 0, 'Heart Disease': 1, 'Infection': 2, 'Injury': 3}
A1C_Result: {'Abnormal': 0, 'Normal': 1, nan: 2}
Readmitted: {'No': 0, 'Yes': 1}

hospital4.py:

Task: 
Data normalization.

Description: 
This script applies MinMax scaling to normalize the numerical columns of the dataset. 
The normalized data is saved to a CSV file.

hospital5.py:

Task: 
Handle class imbalance with SMOTE.

Description: 
This script handles the class imbalance problem in the dataset using Synthetic Minority Over-sampling Technique (SMOTE) to balance the Readmitted classes.

Class distribution before balancing: Counter({0.0: 5054, 1.0: 4946})
Class distribution after balancing: Counter({0.0: 5054, 1.0: 5054})

hospital6.py:

Task:
Train a Random Forest classifier.

Description:
This script trains a Random Forest classifier on the balanced dataset.
It evaluates model performance with a classification report and also plots the feature importance.

Feature Importances:
                  Feature  Importance
0              Patient_ID    0.166758
5      Num_Lab_Procedures    0.147415
1                     Age    0.139030
6         Num_Medications    0.121584
10          Num_Diagnoses    0.082594
7   Num_Outpatient_Visits    0.058067
9    Num_Emergency_Visits    0.057750
8    Num_Inpatient_Visits    0.057183
4               Diagnosis    0.050865
3          Admission_Type    0.040200
2                  Gender    0.039707
11             A1C_Result    0.038846

hospita7.py:

Task: 
Discarding unwanted columns

Description: 
This script performs the columns with the least importance to the target column.

hospital8.py:

Task: 
Again performing the feature importance

Description: 
This script performs feature importance after discarding unwanted columns.

hospital9.py:

Task: 
Handle outliers

Description: 
This script identifies the outliers and removes it.

![image](https://github.com/user-attachments/assets/30e0f568-83fe-4131-85c0-ee3b537a7991)

![image](https://github.com/user-attachments/assets/929c6cf8-9048-46b8-8d5f-7ca423b21d78)

hospital10.py:

Task: 
Handle missing values using IterativeImputer (MICE).

Description: 
This script performs multivariate imputation on missing values in the dataset using the IterativeImputer. 
The processed dataset is saved for further analysis.


hospital11.py:

Task: 
Exploratory Data Analysis (EDA).

Description: 
This script performs data exploration, including boxplots for numerical features against the target and a correlation heatmap to visualize the relationships between features.

![image](https://github.com/user-attachments/assets/baba573e-583a-4998-bd80-98dfd4afb5f6)

![image](https://github.com/user-attachments/assets/876d3094-12a7-4bcb-8b1f-5734da468254)

hospital12.py:

Task: 
Final model training and predictions.

Description: 
This script trains a final Random Forest model on the resampled data, generates predictions, and evaluates the model using confusion matrices and classification reports. The script also saves the dataset with predictions included.

xgb.py:

Task: 
Train an XGBoost classifier on the hospital readmission dataset and fine-tune the model using a GridSearch approach.

Description: 
Load and split the dataset into training and test sets.
Initialize the classifier with parameters obtained from previous grid search.
Train the classifier using early stopping.
Evaluate model performance using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC for both the training and test sets.
Display confusion matrices for both the base model and the fine-tuned model.
Conduct additional fine-tuning by focusing on reg_alpha and reg_lambda.

Test Set:
Accuracy: 0.8534
Precision: 0.8549
Recall: 0.8445
F1 Score: 0.8510
ROC-AUC: 0.8410
Training Set:
Accuracy: 0.8591
Precision: 0.8642
Recall: 0.8563
F1 Score: 0.8579
ROC-AUC: 0.9236
Confusion Matrix:
[[1277  238]
 [ 262 1253]]

xgbv.py:

Task: 
Similar to xgb.py, but this script also incorporates saving the trained model to a pickle file and loading it back for inference or further use.

Description: 
Load and split the dataset.
Train the model using similar parameters as xgb.py and early stopping.
Save the trained model to a file using the pickle module.
Load the model from the saved pickle file and make predictions on the test set.
Evaluate model performance using the same metrics (accuracy, precision, recall, F1 score, ROC-AUC).

--- Test Set Performance ---
Accuracy: 0.8558
Precision: 0.8542
Recall: 0.8583
F1 Score: 0.8562
ROC-AUC: 0.8626

--- Training Set Performance ---
Accuracy: 0.8495
Precision: 0.8549
Recall: 0.8418
F1 Score: 0.8483
ROC-AUC: 0.9151

Confusion Matrix:
[[1077  185]
 [ 179 1084]]

![1](https://github.com/user-attachments/assets/21f1d006-29e4-47be-a7bf-6ed89383c47d)

![2](https://github.com/user-attachments/assets/905c6b39-511e-44b4-bc32-43d5df0b6689)

Data Files

synthetic_hospital_readmissions_data.csv: The original dataset containing features related to patient readmissions.

synthetic_hospital_readmissions_data2.csv: The dataset after label encoding of categorical features.

synthetic_hospital_readmissions_data3.csv: The dataset after normalization of numerical columns.

synthetic_hospital_readmissions_data4.csv: The dataset after balancing with SMOTE.

synthetic_hospital_readmissions_data7.csv: The dataset after missing value imputation using IterativeImputer.

synthetic_hospital_readmissions_data8.csv: The dataset after model predictions are added.

Installation

Prerequisites

Python 3.8+

Required libraries:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

Usage

To run the scripts, execute them in the following sequence to ensure the correct preprocessing and model training pipeline is followed:

Preprocessing and Data Cleaning: 
Start with hospital1.py, hospital2.py, hospital3.py, hospital4.py, and hospital5.py to clean, normalize, and balance the dataset.

Exploratory Data Analysis:
Run hospital11.py to visualize and explore the dataset's features.

Model Training and Evaluation:
Use hospital6.py and hospital12.py for training machine learning models and generating predictions.

Each script is modular and can be adjusted according to the needs of the analysis.

Results

The model demonstrates strong performance across both the test and training sets. 
For the test set, the accuracy is 85.58%, with a precision of 85.42%, recall of 85.83%, and an F1 score of 85.62%, indicating a balanced capability in identifying true positives while minimizing false positives and negatives. 
The ROC-AUC score of 0.8626 suggests that the model performs well at distinguishing between classes. 
On the training set, the model achieves an accuracy of 84.95%, a precision of 85.49%, recall of 84.18%, and an F1 score of 84.83%. 
The ROC-AUC for the training set is notably higher at 0.9151, showing slightly better separation of classes during training. 
The confusion matrix reveals that out of 2,525 predictions, 1,077 true positives and 1,084 true negatives were correctly classified, with 179 false negatives and 185 false positives.

Power Bi

![image](https://github.com/user-attachments/assets/7b81203f-9347-4890-ba2b-947f59c89ea5)
