import numpy as np
from typing import Self
import pandas as pd

"""
This is a suggested template and you do not need to follow it. You can change any part of it to fit your needs.
There are some helper functions that might be useful to implement first.
At the end there is some test code that you can use to test your implementation on synthetic data by running this file.
"""
data=pd.read_csv('coffee_data2.csv')
# data=pd.read_csv('wine_dataset_small.csv')
# cat=data

np_array=data.to_numpy()

X, y = np_array[:, :-1], np_array[:, -1]


# setet=set()
# columns =[np_array[:, i].tolist() for i in range (np_array.shape[1])]


def count(y: np.ndarray) -> np.ndarray:
    proportion=[]
    # columns =[np_array[:, i].tolist() for i in range (np_array.shape[1])]

    unique, counts = np.unique(y, return_counts=True)
    
    # Calculate proportions
    proportions = counts / len(y)
    
    # Sort proportions by the labels
    sorted_proportions = proportions[np.argsort(unique)]
    
    return sorted_proportions

    """
    Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """


# print(count([1, 1, 2, 2, 4, 4, 3, 3, 3]))
# print(count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])))

def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """
    return (1-sum(count(y)**2))


# print(gini_index(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])))

def entropy(y: np.ndarray) -> float:
    """
    Return the entropy of a given NumPy array y.
    """
    return -sum(count(y) * np.log2(count(y))) 
      # Remove this line when you implement the function

# print(entropy(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])))

def split(x: np.ndarray, value: float) -> np.ndarray:
    """
    Return a boolean mask for the elements of x satisfying x <= value.
    Example:
        split(np.array([1, 2, 3, 4, 5, 2]), 3) -> np.array([True, True, True, False, False, True])
    """
    arr=x <= value
    return arr

#print(split(np.array([1, 2, 3, 4, 5, 2]), 3))
        


def most_common(y: np.ndarray) -> int:
    """
    Return the most common element in y.
    Example:
        most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])) -> 4
    """
    unique, counts = np.unique(y, return_counts=True)

# Get the index of the most frequent element
    most_common_index = np.argmax(counts)

    # Get the most frequent element
    most_common_element = unique[most_common_index]

    return most_common_element
    

# print(most_common(np.array([1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])))


class Node:
    """
    A class to represent a node in a decision tree.
    If value != None, then it is a leaf node and predicts that value, otherwise it is an internal node (or root).
    The attribute feature is the index of the feature to split on, threshold is the value to split at,
    and left and right are the left and right child nodes.
    """

    def __init__(
        self,
        feature: int = 0,
        threshold: float = 0.0,
        left: int | Self | None = None,
        right: int | Self | None = None,
        value: int | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        # Return True iff the node is a leaf node
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth: int | None = None, criterion: str = "entropy") -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.root = self.build_tree(X, y)
        """
        This functions learns a decision tree given (continuous) features X and (integer) labels y.
        """
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])


    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
        # Create a leaf node with the most common class
            return Node(value=most_common(y))
        
        # Find the best feature and threshold to split
        best_split = self.find_best_split(X, y)
        
        if not best_split:
            return Node(value=most_common(y))
        
        # Split the data into left and right branches
        left_idxs, right_idxs = best_split['left_idxs'], best_split['right_idxs']
        left_child = self.build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self.build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        # Return a decision node
        return Node(feature=best_split['feature'], threshold=best_split['threshold'], 
                    left=left_child, right=right_child)

    
    
    def find_best_split(self, X, Y):
        best_split = {}
        best_gain = -1
        n_samples, n_features = X.shape

        # Loop through each feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            # Loop through unique feature values to find the best split
            for threshold in unique_values:
                # Split data into left and right branches
                left_idxs = np.where(feature_values <= threshold)[0]
                right_idxs = np.where(feature_values > threshold)[0]
                
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                # Calculate the gain
                gain = self._calculate_gain(y, left_idxs, right_idxs)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'left_idxs': left_idxs,
                        'right_idxs': right_idxs
                    }

        return best_split if best_gain > 0 else None


    
    def _traverse_tree(self, x: np.ndarray, node: Node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        
    def _calculate_gain(self, y: np.ndarray, left_idxs: np.ndarray, right_idxs: np.ndarray) -> float:
        # Get the samples for the left and right splits
        y_left = y[left_idxs]
        y_right = y[right_idxs]
        
        # If using 'gini' as criterion
        if self.criterion == 'gini':
            parent_impurity = gini_index(y)  # Impurity of the parent node
            left_impurity = gini_index(y_left)
            right_impurity = gini_index(y_right)
        
        # If using 'entropy' as criterion
        elif self.criterion == 'entropy':
            parent_impurity = entropy(y)  # Entropy of the parent node
            left_impurity = entropy(y_left)
            right_impurity = entropy(y_right)
        
        # Calculate the weighted impurity of the children
        num_left = len(y_left)
        num_right = len(y_right)
        num_total = len(y)
        
        weighted_avg_child_impurity = (num_left / num_total) * left_impurity + (num_right / num_total) * right_impurity
        
        # Calculate information gain
        gain = parent_impurity - weighted_avg_child_impurity
        
        return gain

    def print_tree(self, node=None, depth=0):
        """
        Recursively print the decision tree structure.
        """
        if node is None:
            node = self.root

        # Print the leaf node
        if node.is_leaf():
            print(f"{'|   ' * depth}Predict: {node.value}")
        else:
            # Print the decision node
            print(f"{'|   ' * depth}Feature {node.feature} <= {node.threshold}")
            
            # Print the left subtree
            print(f"{'|   ' * depth}--> Left:")
            self.print_tree(node.left, depth + 1)
            
            # Print the right subtree
            print(f"{'|   ' * depth}--> Right:")
            self.print_tree(node.right, depth + 1)
    

if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # seed = 0

    #np.random.seed(seed)

    # X, y = make_classification(
    #     n_samples=100, n_features=10, random_state=seed, n_classes=2
    #)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )

    # Expect the training accuracy to be 1.0 when max_depth=None
    rf = DecisionTree(max_depth=None, criterion="entropy")
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")

    
    
