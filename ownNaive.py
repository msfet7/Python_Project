import numpy as np
import pandas as pd

def train_naive_bayes(X_train, y_train):
    n = len(X_train)  # liczba próbek
    target = set(y_train)  # unikalne targety
    feature_count = X_train.shape[1]  # liczba cech
    target_counts = {cls: 0 for cls in target}


    # średnia i warjacja potrzebna do gaussa
    feature_means = {cls: [0] * feature_count for cls in target}
    feature_variances = {cls: [0] * feature_count for cls in target}

    for i in range(n):
        cls = y_train.iloc[i]  
        target_counts[cls] += 1
        for j in range(feature_count):
            feature_means[cls][j] += X_train.iloc[i, j]  

    # średnia cech w każdym targecie
    for cls in target:
        for j in range(feature_count):
            feature_means[cls][j] /= target_counts[cls]

    # warjacja dla każdego targetu
    for i in range(n):
        cls = y_train.iloc[i]  
        for j in range(feature_count):
            diff = X_train.iloc[i, j] - feature_means[cls][j]  
            feature_variances[cls][j] += diff ** 2

    for cls in target:
        for j in range(feature_count):
            feature_variances[cls][j] /= target_counts[cls]

    # prawdopodobieństwo każdego targetu
    target_probabilities = {cls: target_counts[cls] / n for cls in target}

    # zwraca prawdopodobieństwo, średnią i wariację
    # przydadzą się do przewidywań
    return target_probabilities, feature_means, feature_variances


def predict_naive_bayes(X_test, target_probabilities, feature_means, feature_variances):
    predictions = []

    for i in range(len(X_test)):  
        x = X_test.iloc[i].values  
        target_scores = {}
        for cls in target_probabilities:
            score = np.log(target_probabilities[cls])

            for j in range(len(x)):
                # gauss
                mean = feature_means[cls][j]
                variance = feature_variances[cls][j]
                if variance > 0:  
                    score += -0.5 * np.log(2 * np.pi * variance)
                    score += -0.5 * ((x[j] - mean) ** 2) / variance

            target_scores[cls] = score

        # Wybieramy klasę o największym score
        predictions.append(max(target_scores, key=target_scores.get))

    return predictions

def useOwnModel(x_train, y_train, Test):
    # trening modelu
    targetProp, means, variances = train_naive_bayes(x_train, y_train)
    # użycie modelu
    predictions = predict_naive_bayes(Test, targetProp, means, variances)
    # przewidywania, używane do badania dokładności
    return predictions
