import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare the dataset
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Define the autoencoder model
input_dim = X_train.shape[1]

encoder = tf.keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=[input_dim]),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
])

decoder = tf.keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=[32]),
    layers.Dense(128, activation="relu"),
    layers.Dense(input_dim)
])

autoencoder = tf.keras.Sequential([encoder, decoder])
autoencoder.compile(loss="mean_squared_error", optimizer="adam")

# Prepare the data for visualization
def plot_loss(history):
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/03_06_loss_plot.png')  # Save first

# Initialize history variable
history = None

# Training placeholder (to be completed in the end code)