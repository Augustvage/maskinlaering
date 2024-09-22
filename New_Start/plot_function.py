import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def plot(data):
    #Finds features
    features = [col for col in data.columns if col != 'type']
    #Findes name of the column that decides if it is type 1 or 2 in the dataset (0, 1)
    category=data.columns[-1]
    #Sets the two types in a list like this ->[0, 1]
    types=data[category].unique()
    #Plots each feature with two keys (0, 1)
    #This will give histogram with a density scale showing the frequency of a value in a frequency. 
    for feature in features:
        plt.figure(figsize=(10, 6))
        for type in types:
            subset = data[data[category] == type]
            plt.hist(subset[feature], bins=30, alpha=0.5, label=f'Type {type}', density=True, edgecolor='black')
        plt.title(f'Distribution of {feature} by Wine Type')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.show()

def test(model_class, X_train, X_test, y_train, y_test, best_params):
    try:
        model = model_class(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params['max_depth'],
            criterion=best_params['criterion'],
            max_features=best_params['max_features']
        )
    except Exception:
        model=model_class(
            max_depth=best_params['max_depth'],
            criterion=best_params['criterion']
        )

        model.fit(X_train, y_train)
        train_accuracy=accuracy_score(y_train, model.predict(X_train))
        test_accuracy=accuracy_score(y_test, model.predict(X_test))
        print(f"Training accuracy: {train_accuracy}")
        print(f"Test accuracy: {test_accuracy}")

    
    


