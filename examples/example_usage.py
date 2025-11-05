"""
Example usage of Logistic Regression implementation
Demonstrates how to use the LogisticRegression class with a sample dataset.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from logistic_regression import LogisticRegression


def generate_sample_data(n_samples=100, random_state=42):
    """
    Generate a simple binary classification dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : array, shape (n_samples, 2)
        Feature matrix
    y : array, shape (n_samples,)
        Binary labels (0 or 1)
    """
    np.random.seed(random_state)
    
    # Generate class 0 samples (centered around (2, 2))
    n_class_0 = n_samples // 2
    X_class_0 = np.random.randn(n_class_0, 2) + np.array([2, 2])
    y_class_0 = np.zeros(n_class_0)
    
    # Generate class 1 samples (centered around (5, 5))
    n_class_1 = n_samples - n_class_0
    X_class_1 = np.random.randn(n_class_1, 2) + np.array([5, 5])
    y_class_1 = np.ones(n_class_1)
    
    # Combine
    X = np.vstack([X_class_0, X_class_1])
    y = np.hstack([y_class_0, y_class_1])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Labels
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def main():
    """Main function demonstrating logistic regression usage"""
    
    print("=" * 60)
    print("Logistic Regression Example")
    print("=" * 60)
    print()
    
    # Generate sample data
    print("1. Generating sample dataset...")
    X, y = generate_sample_data(n_samples=200, random_state=42)
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of class 0 samples: {np.sum(y == 0)}")
    print(f"   Number of class 1 samples: {np.sum(y == 1)}")
    print()
    
    # Split into training and test sets
    print("2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training set size: {len(X_train)}")
    print(f"   Test set size: {len(X_test)}")
    print()
    
    # Create and train the model
    print("3. Training Logistic Regression model...")
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000, random_state=42)
    model.fit(X_train, y_train)
    print(f"   Initial loss: {model.losses[0]:.4f}")
    print(f"   Final loss: {model.losses[-1]:.4f}")
    print(f"   Learned weights: {model.weights}")
    print(f"   Learned bias: {model.bias:.4f}")
    print()
    
    # Make predictions
    print("4. Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f"   Sample training predictions: {y_pred_train[:10]}")
    print(f"   Sample test predictions: {y_pred_test[:10]}")
    print()
    
    # Evaluate the model
    print("5. Evaluating model performance...")
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"   Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()
    
    # Predict probabilities for a few test samples
    print("6. Probability predictions for first 5 test samples:")
    probabilities = model.predict_proba(X_test[:5])
    for i in range(5):
        print(f"   Sample {i+1}: Features={X_test[i]}, "
              f"Predicted Probability={probabilities[i]:.4f}, "
              f"Predicted Class={y_pred_test[i]}, "
              f"True Class={int(y_test[i])}")
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
