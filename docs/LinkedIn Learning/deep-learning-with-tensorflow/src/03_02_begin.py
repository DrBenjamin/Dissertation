# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fetching the California Housing dataset
housing = fetch_california_housing()

housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
housing_df["Target"] = housing.target

# Visualizing the data
# Displaying the first rows of the dataset
print(housing_df.head())

# Visualizing the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(housing_df["Target"], bins=50, kde=True)
plt.title("Distribution of Target Variable")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.savefig("output/03_02_target_distribution.png")
plt.show()

# Pairplot of the features and target variable
sns.pairplot(housing_df)
plt.savefig("output/03_02_pairplot.png")

# Splitting the data into training, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# Standardizing the data using StandardScaler
scaler = StandardScaler()

# Fitting the scaler on the training data and transforming the validation and test data
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Data preprocessing will be further improved and will be ready to be fed into a TensorFlow model.
# We will define and train our model in the next steps.
