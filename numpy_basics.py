# NumPy Basics - Creating and Manipulating Arrays for AI/ML

import numpy as np

# Creating a 1D array
array_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:")
print(array_1d)

# Creating a 2D array
array_2d = np.array([[1, 2, 3], [56, 78, 98], [25, 879, 987]])
print("\n2D Array:")
print(array_2d)

# Checking the type of array_2d
print("\nType of array_2d:", type(array_2d))

# Creating a 3x3 array filled with zeros
arr_zeros = np.zeros((3, 3), int)
print("\n3x3 Zero Matrix:")
print(arr_zeros)

# Creating a 3x3 array filled with ones
arr_ones = np.ones((3, 3), float)
print("\n3x3 Ones Matrix:")
print(arr_ones)

# Creating a 3x3 identity matrix
identity_matrix = np.identity(3, int)
print("\n3x3 Identity Matrix:")
print(identity_matrix)

# Creating an array using np.arange (start, stop, step)
kk = np.arange(10, 30, 5)
print("\nArray using arange (10 to 30 with step 5):")
print(kk)

# Creating an array using np.linspace (start, stop, num)
arr_linspace = np.linspace(0, 2, 9)
print("\nArray using linspace (0 to 2 with 9 values):")
print(arr_linspace)

# Example: Reshaping an array
reshaped_arr = np.arange(1, 10).reshape(3, 3)
print("\nReshaped 3x3 array from 1 to 9:")
print(reshaped_arr)

# Example: Random number generation
random_arr = np.random.rand(3, 3)
print("\n3x3 Random Matrix:")
print(random_arr)

# Example: Generating a random integer array
random_int_arr = np.random.randint(1, 100, (3, 3))
print("\n3x3 Random Integer Matrix:")
print(random_int_arr)

# Example: Finding shape and size of an array
print("\nShape of array_2d:", array_2d.shape)
print("Size of array_2d:", array_2d.size)

# Example: Transposing an array
transposed_arr = array_2d.T
print("\nTransposed 2D Array:")
print(transposed_arr)

# Example: Basic Mathematical Operations
arr1 = np.array([10, 20, 30])
arr2 = np.array([1, 2, 3])
print("\nElement-wise Addition:", arr1 + arr2)
print("Element-wise Multiplication:", arr1 * arr2)
print("Element-wise Square:", arr1 ** 2)

# Example: Finding maximum and minimum values in an array
print("\nMax value in array_2d:", np.max(array_2d))
print("Min value in array_2d:", np.min(array_2d))

# Example: Mean, Sum, and Standard Deviation
print("\nMean of array_2d:", np.mean(array_2d))
print("Sum of array_2d:", np.sum(array_2d))
print("Standard Deviation of array_2d:", np.std(array_2d))

# AI/ML Data Preprocessing Examples

# Normalizing an array (scaling values between 0 and 1)
data = np.array([50, 20, 80, 100, 30])
norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
print("\nNormalized Data:")
print(norm_data)

# Standardizing an array (mean = 0, std = 1)
standardized_data = (data - np.mean(data)) / np.std(data)
print("\nStandardized Data:")
print(standardized_data)

# Generating a dataset for AI/ML
features = np.random.rand(5, 3)  # 5 samples, 3 features each
labels = np.random.randint(0, 2, 5)  # Binary classification labels
print("\nGenerated Features:")
print(features)
print("\nGenerated Labels:")
print(labels)

# Splitting a dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print("\nTraining Set:")
print(X_train)
print("\nTesting Set:")
print(X_test)

