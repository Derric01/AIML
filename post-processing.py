import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

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

# Encoding Categorical Data
label_encoder = LabelEncoder()
df['Purchased'] = label_encoder.fit_transform(df['Purchased'])

one_hot_encoder = OneHotEncoder(drop='first')
encoded_cities = one_hot_encoder.fit_transform(df[['City']]).toarray()
df = df.drop('City', axis=1)
df = pd.concat([df, pd.DataFrame(encoded_cities, columns=one_hot_encoder.get_feature_names_out(['City']))], axis=1)

# Feature Scaling
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(df.drop(columns=['Purchased']))

# Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_poly)

# Train-Test Split
X = df.drop('Purchased', axis=1)
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Selection using Mutual Information
selector = SelectKBest(mutual_info_classif, k=2)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# Model Training and Evaluation
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Gradient Boosting': GradientBoostingClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter Tuning for Gradient Boosting
param_grid = {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 0.2]}
grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("\nBest Gradient Boosting Parameters:", grid_search.best_params_)

# SHAP Feature Importance
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, show=False)

# LIME Explainability
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=list(X.columns), class_names=['No', 'Yes'], mode='classification')
exp = explainer.explain_instance(X_test[0], rf.predict_proba)
exp.show_in_notebook()

print("\nAdvanced Feature Engineering, Model Training, and Explainability Complete!")
