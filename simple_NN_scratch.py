import numpy as np
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])        # shape: (4, 2)

y = np.array([[0],[0],[0],[1]])  # shape: (4, 1)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)
rng = np.random.default_rng(42)

W1 = rng.normal(size=(2,2))   # input→hidden weights, shape: (in=2, hidden=2)
b1 = np.zeros((1,2))          # hidden bias, shape: (1, 2)

W2 = rng.normal(size=(2,1))   # hidden→output weights, shape: (hidden=2, out=1)
b2 = np.zeros((1,1))          # output bias, shape: (1, 1)

lr = 0.1     # learning rate (step size for weight updates)
epochs = 5000

for epoch in range(epochs):
    # ---- Forward Pass ----
    z1 = np.dot(X, W1) + b1        # (4×2) = (4×2)@(2×2) + (1×2) broadcast
    a1 = sigmoid(z1)               # apply activation, shape stays (4×2)

    z2 = np.dot(a1, W2) + b2       # (4×1) = (4×2)@(2×1) + (1×1) broadcast
    a2 = sigmoid(z2)               # output, shape (4×1)

    loss = np.mean((y - a2) ** 2)

    # output layer gradients
    d_a2 = (a2 - y)                      # d(MSE)/d(a2)
    d_z2 = d_a2 * sigmoid_derivative(a2) # d(MSE)/d(z2) = d(MSE)/d(a2) * d(a2)/d(z2)

    d_W2 = np.dot(a1.T, d_z2)            # (2×4)@(4×1) → (2×1)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)  # (1×1)

    d_a1 = np.dot(d_z2, W2.T)            # (4×1)@(1×2) → (4×2)
    d_z1 = d_a1 * sigmoid_derivative(a1) # elementwise, shape (4×2)

    d_W1 = np.dot(X.T, d_z1)             # (2×4)@(4×2) → (2×2)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)  # (1×2)

    W1 -= lr * d_W1
    b1 -= lr * d_b1
    W2 -= lr * d_W2
    b2 -= lr * d_b2

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

preds = sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2)
print("\nPredictions after training:")
print(np.round(preds, 3))
