#util functions that suppoert the transformed HMM
import numpy as np
import matplotlib.pyplot as plt

def get_nonsequential_data(n,d=2,m=1):
	#n, amount of datapoints
	#d, dimension of data point
	#m, amount of modes

	#--generate data in the unit ball--
	#devide unit ball in to m regions randomly

	centers=np.random.multivariate_normal(np.zeros(d), np.diag(np.ones(d)), m)
	data=np.zeros((n,d))
	n_m=int(n/m)

	sep=4
	covariance=np.power(sep*m,-2/d)
	for i in range(0,m-1):
		data[n_m*i:(i+1)*n_m,:]=np.random.multivariate_normal(centers[i,:],np.diag(np.ones(d)*np.power(sep*m,-2/d)),n_m)
	data[n_m*(m-1):,:]=np.random.multivariate_normal(centers[-1,:],np.diag(np.ones(d)*np.power(sep*m,-2/d)),n-n_m*(m-1))
	np.random.shuffle(data)
	# plt.plot(data[:,0], data[:,1], 'b*')
	# plt.plot(centers[:,0], centers[:,1], 'ro')
	# plt.show()


	return data,centers,covariance


