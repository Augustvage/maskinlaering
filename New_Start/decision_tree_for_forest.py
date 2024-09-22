import numpy as np
from typing import Self
import pandas as pd
import sys

#Find unique 'keys' and how many there are of them. Then it devides on the length to find the porportion
def count(y: np.ndarray) -> np.ndarray:
    proportion=[]

    unique, counts = np.unique(y, return_counts=True)
    
    proportions = counts / len(y)

    sorted_proportions = proportions[np.argsort(unique)]
    
    return sorted_proportions



#Calculate gini index with the given formula
def gini_index(y: np.ndarray) -> float:
    return (1-sum(count(y)**2))

#Calculates the entropy 
def entropy(y: np.ndarray) -> float:
    return -sum(count(y) * np.log2(count(y))) 
      # Remove this line when you implement the function

#Not in use, but takes and array and gives an array with boolean values depending if they under or over 'value'.
def split(x: np.ndarray, value: float) -> np.ndarray:
    arr=x <= value
    return arr

        

#Finds the most common by finding the max of the counts
#If there are two numbers with equal apperances it takes the first one in the 'counts' list. 
def most_common(y: np.ndarray) -> int:
    unique, counts = np.unique(y, return_counts=True)

    most_common_index = np.argmax(counts)

    most_common_element = unique[most_common_index]

    return most_common_element

#Finds the integer value of the squareroot of the length of an array. 
def sqrt(X: np.ndarray) -> int:
    n_features= X.shape[1]
    return int(np.sqrt(n_features))

#Finds the integer value of the logarithm of the length of an array. 
def log2(X: np.ndarray)->int:
    n_features=X.shape[1]
    return int(np.log2(n_features))

#Takes the length of an array and subtracts 1.
#We made it to test some different max_features but removed it because of bad results. 
#Possibly because the number of different combinations decreases. 
def minus_one(X: np.ndarray) -> int:
    n_features=X.shape[1]
    return int(n_features-1)


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

#The decision tree implementation 
class DecisionTree:
    def __init__(self, max_depth: int | None = None, criterion: str = "entropy", max_features=None) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features=max_features

    #Sets the max_features depending on the input and initiate the tree building. 
    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.max_features == 'sqrt':
            self.max_features = sqrt(X)
        elif self.max_features == 'log2':
            self.max_features = log2(X)
        elif self.max_features is None:
            self.max_features = X.shape[1]
        elif self.max_features == 'minus_one':
            self.max_features = minus_one(X)

        self.root = self.build_tree(X, y)
        
    #Initiate the traverse tree so it can travel through each node.
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        return np.array([self.traverse_tree(x, self.root) for x in X])

    #This is the part where we start implementing the ID3 algorithm by recursivly building a tree
    #with two base cases which stops the recurssion. (The base cases are stated in the task)
    #If it has not yet reached the first base case it finds the best split with the best_split function.
    #And if it has not reached the second it add the two new leafes on the old one.
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

    
    #This is the function that the tree builder use decide how to split the array into different nodes. 
    #It uses the calculate gain function to decide the gain of each possible split and chooses the best one. 
    #The gain meassure is decided by the criterion input, and the tree might change depending on what meassure you choose. 
    def find_best_split(self, X: np.ndarray, y: np.ndarray):
        best_split = {}
        best_gain = -1
        n_samples, n_features = X.shape

        # Randomly select a subset of features to consider for this split
        # This should be a list/array of indices
        features_indices = np.random.choice(n_features, self.max_features, replace=False)

        # Loop through each feature in the selected subset
        for feature_idx in features_indices:
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
                gain = self.calculate_gain(y, left_idxs, right_idxs)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'left_idxs': left_idxs,
                        'right_idxs': right_idxs
                    }

        return best_split if best_gain > 0 else None


    #The traverse tree function we used earlier. 
    #Just a function for going through each node in the tree. 

    def traverse_tree(self, x: np.ndarray, node: Node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

     #The calculate gain function. 
     # Calculates gain of a split. You can explain is as the "messiness" of the subsets.    
    def calculate_gain(self, y: np.ndarray, left_idxs: np.ndarray, right_idxs: np.ndarray) -> float:
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

    #A print tree function. It's made to be able to see the tree visually.
    #It is not necesarry for the code to work. 
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
