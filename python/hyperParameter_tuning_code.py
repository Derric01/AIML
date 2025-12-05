from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np
from skopt import BayesSearchCV
from tpot import TPOTClassifier

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Define model
rf = RandomForestClassifier()

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)
print("Best Hyperparameters (Grid Search):", grid_search.best_params_)

# Random Search
param_dist = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': [None, 10, 20],
    'min_samples_split': np.arange(2, 11)
}
random_search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=5, scoring='accuracy')
random_search.fit(X, y)
print("Best Hyperparameters (Random Search):", random_search.best_params_)

# Bayesian Optimization
bayes_search = BayesSearchCV(rf, param_dist, n_iter=30, cv=5, scoring='accuracy')
bayes_search.fit(X, y)
print("Best Hyperparameters (Bayesian Optimization):", bayes_search.best_params_)

# Genetic Algorithm (TPOT)
pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5, scoring='accuracy')
pipeline_optimizer.fit(X, y)
print("Best Model (Genetic Algorithm):", pipeline_optimizer.fitted_pipeline_)
