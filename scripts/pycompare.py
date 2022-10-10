import numpy as np

def dot(A, B):
    a11, a12, a21, a22 = A.ravel()
    b11, b12, b21, b22 = B.ravel()
    h1 = (a21 - a22) * b12
    h2 = (a11 + a21 - a22) * (b12 + b21 + b22)
    h3 = (a11 - a12 + a21 - a22) * (b21 + b22)
    h4 = a12 * b21
    h5 = (a11 + a21) * (b11 + b12 + b21 + b22)
    h6 = a11 * b11
    h7 = a22 * (b12 + b22)
    c11 = (h4 + h6)
    c12 = (- h2 + h5 - h6 - h7)
    c21 = (- h1 + h2 - h3 - h4)
    c22 = (h1 + h7)
    C = np.array([c11, c12, c21, c22]).reshape(2, 2).T.astype(int)
    return C

A = np.array([[1000,2000],[3000,4000]])
B = np.array([[3000,4000],[5000,6000]])

normal = A.dot(B)
alphatesor = dot(A, B)

np.array_equal(normal, alphatesor)
