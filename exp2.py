import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# 1. Load dataset (using breast cancer dataset for demonstration)
X, y = load_breast_cancer(return_X_y=True)

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalize the data (a simple standardization)
# Fit the standardization on the training data and apply to both train and test
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0) + 1e-8  # Avoid division by zero

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Define the Logistic Regression model class
class LogisticRegressionModel:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    # Sigmoid activation function
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Fit the model to the data
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights
        self.bias = 0  # Initialize bias

        # Gradient descent loop
        for iteration in range(self.num_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute the cost (Binary Cross-Entropy)
            cost = -np.mean(y * np.log(y_predicted + 1e-8) + (1 - y) * np.log(1 - y_predicted + 1e-8))

            # Backpropagation (gradient calculation)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print cost every 100 iterations
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.num_iterations}, Cost: {cost:.4f}")

    # Predict using the learned model
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

# Instantiate and train the custom Logistic Regression model
lr_model = LogisticRegressionModel(learning_rate=0.001, num_iterations=5000)
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy on Test Set:", accuracy)
