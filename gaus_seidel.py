import numpy as np
# example_mat = np.mat([[4, -2, 1], [3, 6, -2], [-1, -3, 5]])
# example_answ = np.array([-2, 49, -31])
# coeffs = [3, 6, -2]
# for i in range(example_mat.shape[0]):
# every i element in a row
#     print(example_mat[i,:i])

def Gaus_Seidel(mat, answ, iterations):
    coeffs = np.zeros(answ.size)
    for iter in range(iterations):
        coeffs_old = coeffs.copy()
        for i in range(mat.shape[0]):
            coeffs[i] = (answ[i] - np.dot(mat[i,:i], coeffs[:i]) - np.dot(mat[i,(i+1):], coeffs_old[(i+1):])) / mat[i ,i]
    return coeffs

#print(Gaus_Seidel(example_mat, example_answ, 4))