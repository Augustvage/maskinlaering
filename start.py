import numpy as np
from decision_tree_for_forest import Node, DecisionTree
from sklearn.utils import resample
import math
import pandas as pd

data=pd.read_csv('coffee_data.csv')
# data=pd.read_csv('wine_dataset_small.csv')

np_array=data.to_numpy()

X, y = np_array[:, :-1], np_array[:, -1]


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.trees = [DecisionTree(max_depth=self.max_depth, criterion=self.criterion, max_features=self.max_features) for _ in range(self.n_estimators)]
   
   
    def fit(self, X: np.ndarray, y: np.ndarray):
        for tree in self.trees:
            X_sample, y_sample = resample(X, y, n_samples=len(X))  # Bootstrap sampling
            tree.fit(X_sample, y_sample)


    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Ensure predictions are of integer type
        predictions = predictions.astype(int)
        
        # Take majority vote
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_vote

rf = RandomForest(
    n_estimators=15, 
    max_depth=7, 
    criterion="gini", 
    max_features="log2"
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Number of folds
k = 5

# Create StratifiedKFold instance (to keep class distribution in all folds)
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

# To store accuracy for each fold
accuracies = []

# Perform k-fold cross-validation
for train_index, val_index in skf.split(X, y):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # Fit the RandomForest on the current fold's training data
    rf.fit(X_train_fold, y_train_fold)
    
    # Make predictions on the validation set
    y_val_pred = rf.predict(X_val_fold)
    
    # Calculate accuracy for this fold
    fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
    
    # Append accuracy to list
    accuracies.append(fold_accuracy)

# Calculate the average accuracy across all folds
avg_accuracy = np.mean(accuracies)

print(f"K-Fold Cross-Validation Accuracies: {accuracies}")
print(f"Average Cross-Validation Accuracy: {avg_accuracy}")


