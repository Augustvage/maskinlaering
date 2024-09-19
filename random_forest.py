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




# if __name__ == "__main__":
#     from sklearn.metrics import accuracy_score
#     from sklearn.model_selection import train_test_split

#     seed = 0

#     np.random.seed(seed)

#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.3, random_state=seed, shuffle=True
#     )
#     rf = RandomForest(
#         n_estimators=15, max_depth=7, criterion="gini", max_features="log2"
#     )
#     rf.fit(X_train, y_train)

#     train_accuracy = accuracy_score(y_train, rf.predict(X_train))
#     val_accuracy = accuracy_score(y_val, rf.predict(X_val))

#     print(f"Training accuracy: {train_accuracy}")
#     print(f"Validation accuracy: {val_accuracy}")


        
