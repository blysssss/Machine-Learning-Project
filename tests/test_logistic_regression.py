"""
Unit tests for Logistic Regression implementation
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from logistic_regression import LogisticRegression


class TestLogisticRegression(unittest.TestCase):
    """Test cases for LogisticRegression class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create simple linearly separable dataset
        np.random.seed(42)
        self.X_train = np.array([
            [1, 2], [2, 3], [3, 4], [4, 5],  # Class 0
            [5, 6], [6, 7], [7, 8], [8, 9]   # Class 1
        ])
        self.y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        self.X_test = np.array([[2.5, 3.5], [6.5, 7.5]])
        self.y_test = np.array([0, 1])
    
    def test_initialization(self):
        """Test model initialization"""
        model = LogisticRegression(learning_rate=0.1, n_iterations=500, random_state=42)
        self.assertEqual(model.learning_rate, 0.1)
        self.assertEqual(model.n_iterations, 500)
        self.assertEqual(model.random_state, 42)
        self.assertIsNone(model.weights)
        self.assertIsNone(model.bias)
    
    def test_sigmoid(self):
        """Test sigmoid function"""
        model = LogisticRegression()
        
        # Test with scalar
        self.assertAlmostEqual(model._sigmoid(0), 0.5, places=5)
        self.assertAlmostEqual(model._sigmoid(100), 1.0, places=5)
        self.assertAlmostEqual(model._sigmoid(-100), 0.0, places=5)
        
        # Test with array
        z = np.array([0, 1, -1])
        result = model._sigmoid(z)
        expected = np.array([0.5, 0.73105858, 0.26894142])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_fit(self):
        """Test model fitting"""
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Check that weights and bias are initialized
        self.assertIsNotNone(model.weights)
        self.assertIsNotNone(model.bias)
        self.assertEqual(len(model.weights), self.X_train.shape[1])
        
        # Check that losses are recorded
        self.assertEqual(len(model.losses), 1000)
        
        # Check that loss decreases over time (learning is happening)
        self.assertLess(model.losses[-1], model.losses[0])
    
    def test_predict_proba(self):
        """Test probability prediction"""
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        probabilities = model.predict_proba(self.X_test)
        
        # Check shape
        self.assertEqual(len(probabilities), len(self.X_test))
        
        # Check that probabilities are between 0 and 1
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        
        # For linearly separable data, first sample should have low probability
        # and second sample should have high probability
        self.assertLess(probabilities[0], 0.5)
        self.assertGreater(probabilities[1], 0.5)
    
    def test_predict(self):
        """Test class prediction"""
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        predictions = model.predict(self.X_test)
        
        # Check shape
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Check that predictions are binary (0 or 1)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
        # For linearly separable data, predictions should be correct
        np.testing.assert_array_equal(predictions, self.y_test)
    
    def test_score(self):
        """Test accuracy scoring"""
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Test on training data
        train_score = model.score(self.X_train, self.y_train)
        self.assertGreaterEqual(train_score, 0.0)
        self.assertLessEqual(train_score, 1.0)
        
        # For linearly separable data, we should get high accuracy
        self.assertGreater(train_score, 0.8)
        
        # Test on test data
        test_score = model.score(self.X_test, self.y_test)
        self.assertGreaterEqual(test_score, 0.0)
        self.assertLessEqual(test_score, 1.0)
    
    def test_compute_loss(self):
        """Test loss computation"""
        model = LogisticRegression()
        
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2])
        
        loss = model._compute_loss(y_true, y_pred)
        
        # Loss should be a positive number
        self.assertGreater(loss, 0)
        
        # Perfect predictions should give very low loss
        y_pred_perfect = np.array([0.0001, 0.9999, 0.9999, 0.0001])
        loss_perfect = model._compute_loss(y_true, y_pred_perfect)
        self.assertLess(loss_perfect, loss)
    
    def test_random_state(self):
        """Test that random_state ensures reproducibility"""
        model1 = LogisticRegression(learning_rate=0.1, n_iterations=100, random_state=42)
        model1.fit(self.X_train, self.y_train)
        
        model2 = LogisticRegression(learning_rate=0.1, n_iterations=100, random_state=42)
        model2.fit(self.X_train, self.y_train)
        
        # Same random state should give same results
        np.testing.assert_array_almost_equal(model1.weights, model2.weights)
        self.assertAlmostEqual(model1.bias, model2.bias)
    
    def test_different_learning_rates(self):
        """Test that different learning rates produce different results"""
        model_low = LogisticRegression(learning_rate=0.01, n_iterations=100, random_state=42)
        model_low.fit(self.X_train, self.y_train)
        
        model_high = LogisticRegression(learning_rate=0.5, n_iterations=100, random_state=42)
        model_high.fit(self.X_train, self.y_train)
        
        # Different learning rates should produce different final weights
        self.assertFalse(np.allclose(model_low.weights, model_high.weights))


if __name__ == '__main__':
    unittest.main()
