import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import numpy as np

# Data Preprocessing for AI/ML
# -----------------------------------

# 1. Creating a sample dataset
raw_data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'Age': [25, 30, 35, np.nan, 40],
            'Salary': [50000, 60000, np.nan, 80000, 90000],
            'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles']}

df = pd.DataFrame(raw_data)
print("Original DataFrame:")
print(df)

# 2. Handling Missing Values
# ---------------------------
# Filling missing numeric values with mean

print("\nHandling Missing Values:")
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)
print(df)

# 3. Encoding Categorical Data
# ----------------------------
# Label Encoding for categorical variables
label_encoder = LabelEncoder()
df['City_Label'] = label_encoder.fit_transform(df['City'])
print("\nLabel Encoded City Column:")
print(df[['City', 'City_Label']])

# One-Hot Encoding
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(df[['City']])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(['City']))
df = pd.concat([df, one_hot_encoded_df], axis=1)
print("\nOne-Hot Encoded DataFrame:")
print(df)

# 4. Feature Scaling
# ------------------
# Standardization (Mean = 0, Standard Deviation = 1)
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])
print("\nStandard Scaled DataFrame:")
print(df)

# Min-Max Scaling (Values between 0 and 1)
minmax_scaler = MinMaxScaler()
df[['Age', 'Salary']] = minmax_scaler.fit_transform(df[['Age', 'Salary']])
print("\nMin-Max Scaled DataFrame:")
print(df)

# 5. Handling Outliers
# --------------------
# Using IQR (Interquartile Range) Method
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
print("\nDetected Outliers:")
print(outliers)

# Removing Outliers
df = df[(df['Salary'] >= lower_bound) & (df['Salary'] <= upper_bound)]
print("\nDataFrame after Removing Outliers:")
print(df)

# 6. Feature Engineering
# ----------------------
# Creating a New Feature - Age Group
def age_group(age):
    if age < 30:
        return 'Young'
    elif 30 <= age < 40:
        return 'Middle'
    else:
        return 'Senior'

df['Age_Group'] = df['Age'].apply(lambda x: age_group(x))
print("\nDataFrame with Age Group Feature:")
print(df)

# 7. Saving the Processed Data
# ----------------------------
df.to_csv('processed_data.csv', index=False)
