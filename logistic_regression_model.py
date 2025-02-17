import mlx.core as mx

# The input data, a 2D matrix
X = mx.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# The output data, or "labels"
y = mx.array([0, 1, 1, 1])

# For two input features, we need a weight vector of shape (2,) which is a 1D array with 2 elements
w = mx.zeros(2)
# This is a bias term, an additional parameter that allows the model to fit the data better
# by shifting the decision boundary
b = 0.0
# This determines how much the model's parameters (weights and bias) are adjusted during
# each step of the training process. It determines the size of the steps taken towards
# minimizing the loss function
learning_rate = 0.1
# The number of complete passes the model makes through the entire dataset during training.
# During an epoch, the model processes each training example once and updates its parameters
# (weights and biases) based on the computed gradients
num_epochs = 1000

# Maps any real number to the range [0, 1]
def sigmoid(z):
    return 1 / (1 + mx.exp(-z))

# Computes the model prediction.
# We input X as the data
# w as the weights which determine how important each input is
# b for bias to make better guesses
def predict(X, w, b):
    b_array = mx.full((X.shape[0],), b)
    return sigmoid(mx.addmm(b_array, X, w))

# Measures how good or bad the model's predictions are compared to the actual labels
def binary_cross_entropy(y_true, y_pred, eps=1e-8):
    return -mx.mean(
        y_true * mx.log(y_pred + eps) + (1 - y_true) * mx.log(1 - y_pred + eps)
    )

for epoch in range(num_epochs):
    # Forward pass which computes predictions and loss
    y_pred = predict(X, w, b)
    loss = binary_cross_entropy(y, y_pred)

    # Backwards pass which computes gradients manually. This essentially helps us teach
    # the model how wrong it was in a bad prediction, so that it can learn.
    grad = y_pred - y
    # We look at the effect of each input on the wrong guesses and averages these effects
    grad_w = mx.addmm(mx.zeros_like(X.T @ grad), X.T, grad) / X.shape[0]
    # Calculates how much the bias needs to change. It averages the effect of the bias on the wrong guesses
    grad_b = mx.mean(grad)
    # Update our parameters based on the gradients
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # Print progress every 100 epochs.
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss}")

# If the predicted probability is greater than 0.5, it is classified as 1 (true)
# Otherwise, it is classified as 0 (false)
final_preds = predict(X, w, b) > 0.5
print("Final Predictions:", final_preds)

# Calculate the accuracy of the model
accuracy = mx.mean(final_preds == y)
print(f"Accuracy: {accuracy * 100:.2f}%")