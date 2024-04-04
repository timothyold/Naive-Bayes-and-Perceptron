import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

def naive_bayes_vehicle_usage(datasetpath):
    # Load the dataset
    dataset = pd.read_excel(datasetpath)
    
    # Extract and preprocess the data
    data = dataset[['CAR_USE', 'CAR_TYPE', 'OCCUPATION', 'EDUCATION']].dropna()
    le_car_use = LabelEncoder()
    data['CAR_USE'] = le_car_use.fit_transform(data['CAR_USE'])
    le_features = {}
    features = ['CAR_TYPE', 'OCCUPATION', 'EDUCATION']
    for feature in features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        le_features[feature] = le
    
    X = data[features]
    y = data['CAR_USE']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = CategoricalNB(alpha=0.01)
    model.fit(X_train, y_train)
    
    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    misclassification_rate = 1 - accuracy
    
    # Histogram of predicted probabilities for CAR_USE = Private
    proba = model.predict_proba(X_test)[:, le_car_use.transform(['Private'])[0]]
    plt.hist(proba, bins=np.arange(0, 1.05, 0.05), alpha=0.75, color='blue')
    plt.title('Histogram of Predicted Probabilities for CAR_USE = Private')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.show()
    
    # Evaluate specific fictitious profiles
    profiles = pd.DataFrame([
        {'CAR_TYPE': 'Sports Car', 'OCCUPATION': 'Manager', 'EDUCATION': 'Below High Sc'},
        {'CAR_TYPE': 'SUV', 'OCCUPATION': 'Blue Collar', 'EDUCATION': 'PhD'}
    ])
    for feature in features:
        profiles[feature] = le_features[feature].transform(profiles[feature])
    
    profiles_proba = model.predict_proba(profiles)
    print("Probability distributions for specified profiles (Private, Commercial):")
    for index, prob in enumerate(profiles_proba):
        print(f"Profile {index + 1}: Private = {prob[le_car_use.transform(['Private'])[0]]}, Commercial = {prob[le_car_use.transform(['Commercial'])[0]]}")
    
    return {
        'Accuracy': accuracy,
        'Misclassification Rate': misclassification_rate,
    }


result = naive_bayes_vehicle_usage('lab02_dataset_2.xlsx')
print(result)
