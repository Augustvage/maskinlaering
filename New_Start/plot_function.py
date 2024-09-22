import matplotlib.pyplot as plt

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


