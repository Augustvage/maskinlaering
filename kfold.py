from decision_tree import DecisionTree
from random_forest import RandomForest
from sklearn.model_selection import KFold
import pandas as pd
kf=KFold(n_split=5, shuffle=True, random_state=42)

data=pd.read_csv('coffee_data.csv')
# data=pd.read_csv('wine_dataset_small.csv')
# cat=data

np_array=data.to_numpy()

X, y = np_array[:, :-1], np_array[:, -1]


accuracies = []

# Loop over each fold
for train_index, test_index in kf.split(X):
    # Split data into training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize your DecisionTree model (assuming your DecisionTree class is defined)
    tree = DecisionTree(max_depth=5, criterion="entropy")
    
    # Train the model on the training set
    tree.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = tree.predict(X_test)