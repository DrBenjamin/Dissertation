import tensorflow as tf

# Create a TensorFlow tensor of shape (4, 4) filled with random integers between 0 and 9
tensor = tf.random.uniform(shape=(4, 4), minval=0, maxval=10, dtype=tf.int32)

# Normalize the tensor to have values between 0 and 1
# Task: Normalize the `tensor` so that all its values are between 0 and 1
# Hint: You might need to cast the tensor to a different dtype before performing the division
normalized_tensor = None  # Replace with your code

# Compute the mean and standard deviation of the normalized tensor
# Task: Calculate the mean and standard deviation of `normalized_tensor`
mean = None  # Replace with your code
std_dev = None  # Replace with your code

print("Tensor:", tensor.numpy())
print("Normalized Tensor:", normalized_tensor.numpy() if normalized_tensor is not None else "Not calculated yet")
print("Mean:", mean.numpy() if mean is not None else "Not calculated yet")
print("Standard Deviation:", std_dev.numpy() if std_dev is not None else "Not calculated yet")
