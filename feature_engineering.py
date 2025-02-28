import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, RFE, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap

# Example Dataset
data = {
    'Age': [25, 30, 35, np.nan, 40],
    'Salary': [50000, 60000, 70000, 80000, np.nan],
    'City': ['New York', 'Paris', 'London', 'New York', 'Paris'],
    'Purchased': ['Yes', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Handling Missing Values
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

## Encoding Categorical Data
label_encoder = LabelEncoder()
df['Purchased'] = label_encoder.fit_transform(df['Purchased'])

one_hot_encoder = OneHotEncoder(drop='first')
encoded_cities = one_hot_encoder.fit_transform(df[['City']]).toarray()
df = df.drop('City', axis=1)
df = pd.concat([df, pd.DataFrame(encoded_cities, columns=one_hot_encoder.get_feature_names_out(['City']))], axis=1)

# Feature Scaling
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# Feature Selection using SelectKBest
X = df.drop('Purchased', axis=1)
y = df['Purchased']
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)

# Mutual Information Feature Selection
mi_selector = SelectKBest(score_func=mutual_info_classif, k=2)
X_mi = mi_selector.fit_transform(X, y)

# Recursive Feature Elimination (RFE)
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Dimensionality Reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_new)

# Feature Importance using SHAP
rf = RandomForestClassifier()
rf.fit(X, y)
explainer = shap.Explainer(rf, X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, show=False)

print("Final Processed Data:")
print(pd.DataFrame(X_pca))
