import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fetching the California housing dataset
housing = fetch_california_housing()

# Convert to DataFrame for easier visualization
housing_df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
housing_df['Target'] = housing.target

# Visualizing the data
# Displaying the first few rows to see the structure
print(housing_df.head())

# Visualizing the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(housing_df['Target'], bins=50, kde=True)
plt.title('Distribution of House Values')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.savefig("output/03_02_distribution_plot.png")
plt.close()

# Pairplot of the features to understand relationships
sns.pairplot(housing_df)
plt.savefig('output/03_02_feature_pairplot.png')
plt.close()

# Splitting the data into training, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Visualizing the first few scaled training data points
scaled_train_df = pd.DataFrame(data=X_train, columns=housing.feature_names)
scaled_train_df['Target'] = y_train

print(scaled_train_df.head())

# Visualizing the distribution of the scaled features
plt.figure(figsize=(12, 8))
sns.boxplot(data=scaled_train_df.drop(columns=['Target']))
plt.title('Distribution of Scaled Features')
plt.xticks(rotation=45)
plt.savefig('output/03_02_scaled_features_distribution.png')
plt.close()

# Initializing a simple sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(1)
])

# Compiling the model
model.compile(loss="mean_squared_error", optimizer="sgd")

# Training the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# Evaluating the model on the test set
mse_test = model.evaluate(X_test, y_test)
print(f"Mean Squared Error on Test Set: {mse_test}")
