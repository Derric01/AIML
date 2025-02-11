import numpy as np

# Creating a 1D NumPy array
A = np.array([1, 2, 3], int)

# Iterating through array elements and printing with a comma separator
for x in A:
    print(x, end=",")
print()

# Corrected definition of a 2D NumPy array
B = np.array([[1, 3, 4], [384, 34, 0], [33, 5, 6]])

# Iterating through rows of the 2D array
for y in B:
    print(y, end=",")
print()

# Creating 2D NumPy arrays and performing matrix operations
a = np.array(([1, 2], [12, 6]), int)
b = np.array(([12, 34], [43, 43]), int)  # `np.matrix` is deprecated, using `np.array` instead

# Addition of two matrices
print("Matrix Addition:\n", a + b)

# Dot product (Matrix Multiplication)
print("Dot Product of a and b:\n", a.dot(b))
print("Dot Product of b and a:\n", b.dot(a))

# Sum, Product, Min, and Shape of Matrix `a`
print("Sum of elements in a:", np.sum(a))
print("Product of elements in a:", np.prod(a))
print("Minimum element in a:", np.min(a))
print("Shape of a:", np.shape(a))

# Concatenating multiple arrays
a = np.array([1, 2])
b = np.array([34, 2])
c = np.array([2, 45])
print("Concatenated Array:", np.concatenate([a, b, c]))

# Concatenating 2D arrays along different axes
a = np.array([[1, 2], [3, 4]])
b = np.array([[8, 25], [63, 74]])
print("Concatenation along axis 0:\n", np.concatenate((a, b), axis=0))
print("Concatenation along axis 1:\n", np.concatenate((a, b), axis=1))

# Sorting an array
A = np.array([1, 3, 4, -1, 0])
print("Sorted Array:", np.sort(A))

# Extracting the diagonal elements of a matrix
A = np.array([[1, 2], [34, 65]])
print("Diagonal Elements:", A.diagonal())

# Creating a NumPy array using `arange()` and applying power operation
A = np.arange(10) ** 3
print("Cubed Elements:", A)

# Reshaping an array
x = np.arange(0, 10)
print("Original Array:", x)
x = x.reshape(5, 2)
print("Reshaped Array:\n", x)

# Flattening a 2D array
x = np.array([[0, 102, 23], [23, 43, 4], [34, 76, 45]])
print("Flatten using flatten():", x.flatten())
print("Flatten using ravel():", x.ravel())  # More memory efficient

# Creating a NumPy array with a step size
arr_1 = np.arange(10, 30, 5)
arr = np.arange(0, 3, 2)
print("Array with step 5:", arr_1)
print("Array with step 2:", arr)

# ---------------- ADVANCED CONCEPTS ----------------

# 1. Broadcasting: Automatically adjusts array shapes for operations
A = np.array([1, 2, 3])
B = np.array([[1], [2], [3]])
print("Broadcasted Addition:\n", A + B)

# 2. Boolean Masking & Filtering: Extract elements based on conditions
A = np.array([1, 2, 3, 4, 5, 6])
print("Elements greater than 3:", A[A > 3])

# 3. Random Number Generation: Creating random arrays
rand_arr = np.random.rand(3, 3)  # 3x3 matrix with values between 0 and 1
print("Random 3x3 Matrix:\n", rand_arr)

rand_int_arr = np.random.randint(10, 100, (3, 3))  # Random integers between 10 and 100
print("Random Integer 3x3 Matrix:\n", rand_int_arr)
