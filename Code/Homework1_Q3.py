import numpy as np
from numpy import linalg as LA
from numpy import diag
from numpy import zeros
A=np.array([[-5,-5,-1,0,0,0,500,500,100],[0,0,0,-5,-5,-1,500,500,100],
           [-150,-5,-1,0,0,0,30000,1000,200],[0,0,0,-150,-5,-1,12000,400,80],
           [-150,-150,-1,0,0,0,33000,33000,220],[0,0,0,-150,-150,-1,12000,12000,80],
           [-5,-150,-1,0,0,0,500,15000,100],[0,0,0,-5,-150,-1,1000,30000,200]])
print("The A matrix:\n{}".format(A))
A_T=A.transpose()
print("The transpose of the A matrix is: \n{}".format(A_T))
A1=A.dot(A_T)
A2=A_T.dot(A)
#To find the eigen vectors and eigenvalues of the A^T*A and A*A^T.
w1, v1 = LA.eig(A1)
w2, v2 = LA.eig(A2)
#Taking the absolute values of the eigen values to keep them postive
w= abs(w2)
print("The adjusted eigen values : \n{}".format(w))
U=v1
print("The Eigen Vector U is: \n{}".format(U))
#transpose of the V eigen vector matrix
V_T=v2.transpose()
print("Eigen vectors 2\n{}".format(v2))
print("The transpose of Eigen Vector V is: \n{}".format(V_T))
#Sigma Matrix matrix can be formed using the square root of eigenvalues of A^T*A
#Sigma matrix is a diagonal matrix with the squareroot of the eigen values
sig=np.sqrt(w)
print("The eigenvectors considered for the Sigma matrix are: \n{}".format(sig))
#To form the Sigma diagonal matrix
Sigma = zeros((A.shape[0], A.shape[1]))
print(Sigma)
for i in range(len(Sigma)):
    Sigma[i][i]= sig[i]
print("The Sigma Matrix(Diagonal Matrix) is : \n{}".format(Sigma))
#SVD of A= U*Sigma*V'
SVD=U.dot(Sigma.dot(V_T))
print("The SVD of Matrix A is:")
print(SVD)
#To find the Homography matrix
H=v2[:,-1]
H1=np.reshape(H,(3,3))
print("The Homography matrix H is:")
print(H1)
