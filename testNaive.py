import unittest
import pandas as pd
from ownNaive import train_naive_bayes


class TestNaiveBayes(unittest.TestCase):

    def setUp(self):
        # Przygotowanie przykładowych danych
        data = {'feature1': [1, 2, 3, 4],
                'feature2': [5, 6, 7, 8],
                'target': [0, 1, 0, 1]}
        df = pd.DataFrame(data)
        self.X_train = df.drop('target', axis=1)
        self.y_train = df['target']

    def test_target_counts(self):
        # Trenujemy model
        target_probabilities, feature_means, feature_variances = train_naive_bayes(self.X_train, self.y_train)
        
        # Sprawdzamy liczbę próbek w każdej klasie
        expected_target_counts = {0: 2, 1: 2}
        
        for cls in expected_target_counts:
            self.assertEqual(target_probabilities[cls], expected_target_counts[cls] / len(self.X_train))

    def test_feature_means(self):
        # Trenujemy model
        target_probabilities, feature_means, feature_variances = train_naive_bayes(self.X_train, self.y_train)
        
        # Sprawdzamy średnie dla cech w każdej klasie
        expected_means = {
            0: [2.0, 6.0],
            1: [3.0, 7.0]
        }
        
        for cls in expected_means:
            for i in range(len(expected_means[cls])):
                self.assertAlmostEqual(feature_means[cls][i], expected_means[cls][i], places=1)

    def test_single_target(self):
        # Przygotowujemy dane, gdzie wszystkie próbki należą do klasy 0
        data = {'feature1': [1, 2, 3],
                'feature2': [4, 5, 6],
                'target': [0, 0, 0]}
        df = pd.DataFrame(data)
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        # Trenujemy model
        target_probabilities, feature_means, feature_variances = train_naive_bayes(X_train, y_train)
        
        # Prawdopodobieństwo a priori klasy 0 powinno być 1
        self.assertEqual(target_probabilities[0], 1.0)
    
    def test_different_data_sizes(self):
        # Test z mniejszymi danymi
        data_small = {'feature1': [1, 2],
                      'feature2': [4, 5],
                      'target': [0, 1]}
        df_small = pd.DataFrame(data_small)
        X_train_small = df_small.drop('target', axis=1)
        y_train_small = df_small['target']
        
        # Test z większymi danymi
        data_large = {'feature1': list(range(100)),
                      'feature2': list(range(100, 200)),
                      'target': [0]*50 + [1]*50}
        df_large = pd.DataFrame(data_large)
        X_train_large = df_large.drop('target', axis=1)
        y_train_large = df_large['target']
        
        # Trenujemy modele
        target_probabilities_small, feature_means_small, feature_variances_small = train_naive_bayes(X_train_small, y_train_small)
        target_probabilities_large, feature_means_large, feature_variances_large = train_naive_bayes(X_train_large, y_train_large)
        
        self.assertTrue(target_probabilities_small is not None)
        self.assertTrue(target_probabilities_large is not None)

if __name__ == '__main__':
    unittest.main()