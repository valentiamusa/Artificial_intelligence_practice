import numpy as np

# Function to get a matrix from the user
def get_matrix(name):
    rows = int(input(f"Enter the number of rows for {name}: "))
    cols = int(input(f"Enter the number of columns for {name}: "))
    print(f"Enter the elements of {name} row by row, separated by spaces:")
    matrix = []
    for i in range(rows):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if len(row) != cols:
            print("Error: Number of elements in the row does not match the number of columns.")
            return None
        matrix.append(row)
    return np.array(matrix)

# Get two matrices from the user
matrix1 = get_matrix("Matrix 1")
matrix2 = get_matrix("Matrix 2")

# Check if matrices were entered correctly
if matrix1 is None or matrix2 is None:
    print("Matrix input error. Exiting program.")
else:
    # Addition and Subtraction require same shape
    if matrix1.shape == matrix2.shape:
        sum_matrix = matrix1 + matrix2
        diff_matrix = matrix1 - matrix2
        print(f"\nSum of matrices:\n{sum_matrix}")
        print(f"\nDifference of matrices:\n{diff_matrix}")
    else:
        print("\nCannot add or subtract matrices with different shapes.")

    # Multiplication: columns of first = rows of second
    if matrix1.shape[1] == matrix2.shape[0]:
        prod_matrix = np.dot(matrix1, matrix2)
        print(f"\nProduct of matrices:\n{prod_matrix}")
    else:
        print("\nCannot multiply matrices: incompatible dimensions.")
