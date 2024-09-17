import numpy as np
from decision_tree import Node, DecisionTree
from sklearn.utils import resample
import math

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
        self.trees = [DecisionTree(max_depth=self.max_depth, criterion=self.criterion) for _ in range(self.n_estimators)]
   
   
    def fit(self, X: np.ndarray, y: np.ndarray):
        for tree in self.trees:
            X_sample, y_sample = resample(X, y, n_samples=len(X))  # Bootstrap sampling
            tree.fit(X_sample, y_sample)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Take majority vote
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_vote


if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 0

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )
temp=0
temp2=0
dicten={'high_total': None, 'best': None}
for i in range (1, 40):
    for e in range (1, 10):
        rf = RandomForest(
            n_estimators=i, max_depth=e, criterion="entropy", max_features="sqrt"
        )
        rf.fit(X_train, y_train)

        # print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
        # print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
        # print(i, e)


        t=accuracy_score(y_train, rf.predict(X_train))
        v=accuracy_score(y_val, rf.predict(X_val))
        if t+v>temp:
            temp=t+v
            dicten['high_total']=[t, v, i, e]
        if t+v-abs(t-v)>temp2:
            temp2=t+v-abs(t-v)>temp2
            dicten['best']=[t, v, i, e]
        print(dicten)

        
