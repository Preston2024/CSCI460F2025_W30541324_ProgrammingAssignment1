######################################################
#  Student Name: Preston E. Mincey
#  Student ID: W30541324
#  Course Code: CSCI 460 -001  Fall 2025
#  Assignment Due Date: October 14, 2025
#  GitHub Link: https://github.com/Preston2024/CSCI460F2025_W30541324_ProgrammingAssignment1 
######################################################

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

# Load and preprocess data
df = pd.read_csv('bank-full.csv', sep=';')
# Map categorical variables to integers starting at 0
for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']:
    df[col] = pd.factorize(df[col])[0]
# Drop any rows with missing data
df = df.dropna()

# Feature and target split
X = df.drop('y', axis=1).values
y = df['y'].values

def run_experiment(train_size, num_trials=10):
    accuracies = []
    f1s = []
    sss = StratifiedShuffleSplit(n_splits=num_trials, train_size=train_size, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracies.append(acc)
        f1s.append(f1)
    return accuracies, f1s

def main():
    split_fractions = [0.1 * i for i in range(1, 10)]
    results = []
    print("Train Fraction,Trial,Accuracy,F1-Score")
    for frac in split_fractions:
        accuracies, f1s = run_experiment(frac, num_trials=10)
        for i in range(10):
            print(f"{frac:.1f},{i+1},{accuracies[i]:.4f},{f1s[i]:.4f}")
        # Summary statistics
        print(f"{frac:.1f},Mean,{np.mean(accuracies):.4f},{np.mean(f1s):.4f}")
        print(f"{frac:.1f},Std,{np.std(accuracies):.4f},{np.std(f1s):.4f}")

if __name__ == "__main__":
    main()
