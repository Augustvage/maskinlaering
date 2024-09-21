from decision_tree_for_forest import DecisionTree as DT2
from random_forest import RandomForest
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import itertools
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC

def grid_search(model_class, X, y, n_estimators=None, max_depth=None, criterion=None, max_features=None):
    decision_tree=False
    seed = 0
    np.random.seed(seed)
    k = 5
    # Create StratifiedKFold instance (to keep class distribution in all folds)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    # Parameters to grid search over
    params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'criterion': criterion, 'max_features': max_features}

    best_accuracy = 0
    best_params = None

    # Get the product of parameter combinations
    param_combinations = itertools.product(
        params['n_estimators'] or [None], 
        params['max_depth'] or [None], 
        params['criterion'] or [None], 
        params['max_features'] or [None]
    )

    for param_set in param_combinations:
        n_estimators, max_depth, criterion, max_features = param_set
        accuracies = []

        for train_index, val_index in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Initialize the model dynamically using model_class
            try:
                model = model_class(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    criterion=criterion,
                    max_features=max_features
                )
            except Exception:
                decision_tree=True
                model = model_class(
                    max_depth=max_depth,
                    criterion=criterion,
                    max_features=max_features
                )


            # Train the model
            model.fit(X_train_fold, y_train_fold)

            # Evaluate on validation fold
            y_pred = model.predict(X_val_fold)
            accuracy = np.mean(y_pred == y_val_fold)
            accuracies.append(accuracy)

        # Average accuracy for this combination
        avg_accuracy = np.mean(accuracies)

        # Update best accuracy and parameters if the current set is better
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'criterion': criterion,
                'max_features': max_features
            }
    if decision_tree is True:
        del best_params['n_estimators']

    return best_params, best_accuracy

