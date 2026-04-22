from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Load and split the California Housing dataset
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Define a simple sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(
    loss="mean_squared_error",
    optimizer="sgd",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanSquaredError(name='mse')
    ]
)

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# Evaluate the model
test_results = model.evaluate(X_test, y_test, return_dict=True)
print(f"Test Results: {test_results}")

# Visualize training history
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error (MAE)')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['rmse'], label='Training RMSE')
plt.plot(history.history['val_rmse'], label='Validation RMSE')
plt.title('Root Mean Squared Error (RMSE)')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()

plt.tight_layout()

# Save the trained model
model.save('output/03_05_trained_model.h5')

# Load the model back
loaded_model = tf.keras.models.load_model('output/03_05_trained_model.h5')

# Evaluate the loaded model to confirm it's correctly loaded
loaded_test_results = loaded_model.evaluate(X_test, y_test, return_dict=True)
print(f"Loaded Model Test Results: {loaded_test_results}")

# Compare predictions from the original and loaded models
original_preds = model.predict(X_test)
loaded_preds = loaded_model.predict(X_test)

# Assert to verify that the predictions are the same
assert tf.reduce_all(tf.abs(original_preds - loaded_preds) < 1e-5), "Predictions differ between original and loaded models!"

# Visualization of original and loaded model predictions
plt.figure(figsize=(14, 7))
plt.plot(original_preds, label='Original Model Predictions')
plt.plot(loaded_preds, label='Loaded Model Predictions', linestyle='dashed')
plt.title('Original vs Loaded Model Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Value')
plt.legend()

# Save this visualization
plt.savefig('output/03_05_model_comparison.png')
