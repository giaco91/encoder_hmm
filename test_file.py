#test file for the transformed MMM
import matplotlib.pyplot as plt
import numpy as np
#generate data
from utils import *
from gm import *
from gmm import *

data,centers,covariance=get_nonsequential_data(1000,d=2,m=4)
model=GMM(n_mixture=2)
LL=model.fit(data)
print(model.score(data))
samples=model.sample(50)

# print(model.means)
# print(model.weights)
# print(model.covars)
# print(model.invcovars)
# print(model.normalizations)
# print(LL)
plt.plot(samples[:,0],samples[:,1],'r*')
plt.plot(data[:,0], data[:,1], 'b*')
plt.plot(centers[:,0],centers[:,1],'bo')
plt.plot(model.means[:,0],model.means[:,1],'ro')

plt.show()
