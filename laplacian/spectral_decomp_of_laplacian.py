import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


deg_m = np.array([
	[1, 0, 0, 0, 0],
	[0, 4, 0, 0, 0],
	[0, 0, 1, 0, 0],
	[0, 0, 0, 2, 0],
	[0, 0, 0, 0, 2],
])

adj_m = np.array([
	[1, 1, 0, 0, 0],
	[1, 1, 1, 1, 1],
	[0, 1, 1, 0, 0],
	[0, 1, 0, 1, 1],
	[0, 1, 0, 1, 1],
])

lap_m = np.array([
	[0, -1,  0,  0,  0],
	[-1,  3, -1, -1, -1],
	[0, -1,  0,  0,  0],
	[0, -1,  0,  1, -1],
	[0, -1,  0, -1,  1],
])

adj_sm = csr_matrix(adj_m)


print("Eigendecomposition of Laplacian\n")
print(np.linalg.eig(lap_m))
print("\n")

print("Connected Components from Adjacency Matrix\n")
print(connected_components(adj_sm))
