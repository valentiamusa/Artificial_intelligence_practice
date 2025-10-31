import numpy as np

# Function to get a vector from the user
def get_vector(name):
    vector_str = input(f"Enter the elements of {name} separated by spaces: ")
    vector_list = [float(x) for x in vector_str.split()]
    return np.array(vector_list)

# Get two vectors from the user
vector1 = get_vector("Vector 1")
vector2 = get_vector("Vector 2")

# Check if the vectors are the same size
if vector1.shape != vector2.shape:
    print("Error: Vectors must have the same length.")
else:
    # Compute sum
    sum_vector = vector1 + vector2
    # Compute difference
    diff_vector = vector1 - vector2
    # Compute dot product
    dot_product = np.dot(vector1, vector2)

    # Display results
    print(f"\nVector 1: {vector1}")
    print(f"Vector 2: {vector2}")
    print(f"Sum: {sum_vector}")
    print(f"Difference: {diff_vector}")
    print(f"Dot Product: {dot_product}")