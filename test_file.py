#test file for the transformed MMM
import matplotlib.pyplot as plt
import numpy as np
#generate data
from utils import *
from gm import *

data,centers,covariance=get_nonsequential_data(100,d=2,m=3)
model=GM()
LL=model.fit(data)
print(LL)
samples=model.sample(40)
plt.plot(data[:,0], data[:,1], 'b*')
plt.plot(centers[:,0],centers[:,1],'bo')
plt.plot(model.mean[0],model.mean[1],'ro')
plt.plot(samples[:,0],samples[:,1],'r*')
plt.show()
