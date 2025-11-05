# Machine Learning Project - Logistic Regression

A from-scratch implementation of Logistic Regression for binary classification using gradient descent optimization.

## Overview

This project implements Logistic Regression, a fundamental machine learning algorithm for binary classification. The implementation uses:
- **Sigmoid activation function** for probability predictions
- **Binary cross-entropy loss** for optimization
- **Gradient descent** for parameter learning
- Pure **NumPy** implementation (no scikit-learn dependencies)

## Features

- ✅ Complete logistic regression implementation from scratch
- ✅ Configurable learning rate and number of iterations
- ✅ Support for probability and class predictions
- ✅ Built-in accuracy scoring
- ✅ Loss tracking during training
- ✅ Reproducible results with random state
- ✅ Comprehensive unit tests
- ✅ Example usage with sample dataset

## Installation

### Requirements
- Python 3.6+
- NumPy

### Setup
```bash
# Clone the repository
git clone https://github.com/blysssss/Machine-Learning-Project.git
cd Machine-Learning-Project

# Install dependencies
pip install numpy
```

## Usage

### Basic Example

```python
from src.logistic_regression import LogisticRegression
import numpy as np

# Create sample data
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [5, 6], [6, 7], [7, 8], [8, 9]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Create and train the model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([[2.5, 3.5], [6.5, 7.5]])
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

### Running the Example

```bash
python examples/example_usage.py
```

This will:
1. Generate a sample binary classification dataset
2. Split data into training and test sets
3. Train a logistic regression model
4. Display training progress and results
5. Show predictions and evaluation metrics

## API Reference

### LogisticRegression

**Parameters:**
- `learning_rate` (float, default=0.01): Learning rate for gradient descent
- `n_iterations` (int, default=1000): Number of training iterations
- `random_state` (int, default=None): Random seed for reproducibility

**Methods:**
- `fit(X, y)`: Train the model on data X with labels y
- `predict(X)`: Predict binary class labels (0 or 1)
- `predict_proba(X)`: Predict probability estimates
- `score(X, y)`: Calculate accuracy score

**Attributes:**
- `weights`: Learned feature weights
- `bias`: Learned bias term
- `losses`: List of loss values during training

## Testing

Run the unit tests:

```bash
python -m unittest tests/test_logistic_regression.py
```

Or run with verbose output:

```bash
python tests/test_logistic_regression.py -v
```

## Project Structure

```
Machine-Learning-Project/
├── src/
│   ├── __init__.py
│   └── logistic_regression.py    # Main implementation
├── tests/
│   └── test_logistic_regression.py  # Unit tests
├── examples/
│   └── example_usage.py          # Example usage
└── README.md
```

## Algorithm Details

### Logistic Regression

Logistic regression predicts the probability of a binary outcome using the sigmoid function:

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Model:**
```
z = w^T x + b
y_pred = σ(z)
```

**Loss Function (Binary Cross-Entropy):**
```
L = -1/m Σ [y * log(ŷ) + (1-y) * log(1-ŷ)]
```

**Gradient Descent Update:**
```
w = w - α * ∂L/∂w
b = b - α * ∂L/∂b
```

Where:
- w: weights
- b: bias
- α: learning rate
- m: number of samples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.