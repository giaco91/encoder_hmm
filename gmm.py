import numpy as numpy
from distributions import *

class GM():

    def __init__(self,D=None,mean=None,covar=None):
        self.D=None
        self.mean=None
        self.covar=None

    def check_symmetric(self,A):
        symmetric=np.allclose(A, A.T)
        return symmetric
    def check_pos_def(self,A):
        posdef=np.all(np.linalg.eigvals(A) > 1e-4)#not too small eigenvalues
        return posdef


    def fit(self,x):
    	#x is a np array of shape (N,D), where N is the # of data points and D the dimension
    	self.D=x.shape[1]
    	N=x.shape[0]
    	self.mean=np.sum(x,axis=0)/N
    	self.covar=np.zeros((self.D,self.D))
    	x_unbiased=x-self.mean
    	for n in range(0,N):
    		self.covar+=np.outer(x_unbiased[n,:],x_unbiased[n,:])
    	self.covar=self.covar/N
    	self.invcovar=np.linalg.inv(self.covar)
    	self.density_proportionalfactor=np.power(2*np.pi,-self.D/2)*np.power(np.linalg.det(self.covar),-1/2)
    	LL=0
    	for n in range(0,N):
    		LL+=np.log(self.density(x[n,:]))
    	return LL

    def sample(self,n):
    	return np.random.multivariate_normal(self.mean,self.covar,n)

    def density(self,x):
        if x.shape[0]!=self.D:
            raise ValueError('The evaluation point x has not dimension D!')
        return self.density_proportionalfactor*np.exp(-(1/2)*np.einsum('i,i',x-self.mean,np.einsum('ij,j', self.invcovar, x-self.mean)))

