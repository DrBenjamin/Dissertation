import tensorflow as tf

# Create a TensorFlow tensor of shape (4, 4) filled with random integers between 0 and 9
tensor = tf.random.uniform(shape=(4, 4), minval=0, maxval=10, dtype=tf.int32)

# Cast the tensor to float before normalization
tensor = tf.cast(tensor, tf.float32)

# Normalize the tensor to have values between 0 and 1
normalized_tensor = tensor / 9.0

# Compute the mean of the normalized tensor
mean = tf.reduce_mean(normalized_tensor)

# Compute the standard deviation of the normalized tensor
std_dev = tf.math.reduce_std(normalized_tensor)

print("Tensor:", tensor.numpy())
print("Normalized Tensor:", normalized_tensor.numpy())
print("Mean:", mean.numpy())
print("Standard Deviation:", std_dev.numpy())
