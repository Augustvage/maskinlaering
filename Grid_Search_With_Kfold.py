from sklearn.model_selection import KFold, GridSearchCV
import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import itertools


#Grid Search with Kfold-validation.
#This does tune the hyperparamters by "setting up a grid" and test each "cell".
#Each cell has a unique combination. By testing all the accurisies of the different combinations
#Will you find the best one. Because we have Kfold will it be tested by different parts of the dataset
#meaning that we reduce variance because we will get the output that is on average best. 
def grid_search(model_class, X, y, n_estimators=None, max_depth=None, criterion=None, max_features=None):
    decision_tree=False   #Just a variable to destinct between the models since the decision tree don't have n_estimators as a hyperparameter. 
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
            accuracy = accuracy_score(y_val_fold, y_pred)
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


