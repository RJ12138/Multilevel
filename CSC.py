import numpy as np
from math import acos, sin, cos

def is_symmetric(A):
	return (np.abs(A - A.T) <= 1e-9).all()

def create_combinatorial_lap(A):
	if is_symmetric(A):
		D = np.diag(np.sum(A, axis = 1))
		return D - A
	else:
		print('create_laplaciangraph: A should be symmetric')

def create_normalized_lap(A):
	if is_symmetric(A):
		R = np.sum(A, axis = 1)
		R_sqrt = 1/np.sqrt(R)
		D_sqrt[np.isinf(D_sqrt)|np.isnan(D_sqrt)] = 0
		D_sqrt = np.diag(R_sqrt)
		I = np.eye(A.shape[0])

		return I - D_sqrt * A * D_sqrt
	else:
		print('create_laplaciangraph: A should be symmetric')

def create_laplacian(A,lapversion):

	# creates the Laplacian matrix of a graph represented by its adjacency matrix A. 
	# lapversion should be a string equal to 'combinatorial' if you want L=D-A; 
	# or 'normalized' if you want the normalized version of the laplacian L=I-D^(-1/2)*A*D^(-1/2); 
	# where D is the diagonal matrix of the degrees. 
	# A should be symmetrical.

	if lapversion = 'combinatorial':
	    return create_combinatorial_lap(A)
	elif lapversion = 'normalized':   
	    return create_normalized_lap(A)
	else:
		print('Laplacian type should be combinatorial or normalized')

def jackson_cheby_poly_coefficients(a,b,lambda_range,m):

	# That is to compute the m+1 coefficients of the polynomial approximation of 
	# an ideal band-pass between a and b, in between a range of values defined by 
	# lambda_range=[lambda_min,lambda_max]; 
	# ----
	# Output:
	# - CH are the coefficients of the Chebychev polynomials
	# - JCH are the coefficients of the jackson-chebychev polynomials

	# scaling and translation coefficients compared to the classical interval
	# of Chebychev polynomials [-1,1] :
	a1 = (lambda_range[1]-lambda_range[0])/2
	a2 = (lambda_range[0]+lambda_range[1])/2

	# scale the boundaries of the band pass according to lrange:
	a=(a-a2)/a1
	b=(b-a2)/a1

	CH = np.zeros(m+1, dtype=np.float)
	gamma_JACK = np.zeros(m+1, dtype=np.float)
	pi = 3.1415926

	# compute Cheby coef:
	CH[0]=(1/pi)*(acos(a)-acos(b))
	for j in range(1, m+1):
	    CH[j] = (2/(pi*(j-1)))*(sin((j-1)*acos(a))-sin((j-1)*acos(b)))

	# compute Jackson coef:
	alpha=pi/(m+2)
	for j in range(m+1):
	    gamma_JACK(j)=(1/sin(alpha))*((1-(j-1)/(m+2))*sin(alpha)*cos((j-1)*alpha)+(1/(m+2))*cos(alpha)*sin((j-1)*alpha))

	# compute Jackson-Cheby coef:
	JCH = np.multiply(CH, gamma_JACK)

	# to be in adequation with gsp_cheby_op.m :
	JCH[0]=JCH[0] * 2
	CH[0]=CH[0] * 2

	return CH, JCH

def getFourierBasis(L, algo='eigh', k=1, norm = 2):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    # norm = 'l' + str(norm)
    # U = preprocessing.normalize(U,norm = norm ,axis =0)

    return lamb, U