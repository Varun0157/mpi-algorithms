import numpy as np

# Define the size of the matrix
N = 200  # note: test file working for size 750

# Generate a random non-singular square matrix with 2 decimal places
while True:
    matrix = np.random.rand(N, N)
    matrix = np.round(matrix, 2)
    if np.linalg.det(matrix) != 0:
        break

# Write the matrix to random.txt
with open("random.txt", "w") as f:
    f.write(f"{N}\n")
    for row in matrix:
        f.write(" ".join(f"{elem:.2f}" for elem in row) + "\n")

# Calculate the inverse of the matrix, rounded to 2 decimal places
inverse_matrix = np.linalg.inv(matrix)
inverse_matrix = np.round(inverse_matrix, 2)

# Write the inverse matrix to random-opt.txt
with open("random-opt.txt", "w") as f:
    for row in inverse_matrix:
        f.write(" ".join(f"{elem:.2f}" for elem in row))
        f.write(" \n")

print(
    "Matrix and its inverse have been written to random.txt and random-opt.txt respectively."
)
