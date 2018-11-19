#test file for the transformed MMM
import matplotlib.pyplot as plt
import numpy as np
#generate data
from utils import *
from gm import *
from gmm import *
from ed_nn import *

data,centers,covariance=get_nonsequential_data(500,d=2,m=4)
# model=GMM(n_mixture=2)
# LL=model.fit(data)
ednn=EDNN(D=2)
ednn.init_parameters(identity=True)
loss=ednn.learn_double_identity(data)
print('final loss: '+str(loss))
# print(model.score(data))
# samples=model.sample(50)


# plt.plot(samples[:,0],samples[:,1],'r*')

# plt.plot(centers[:,0],centers[:,1],'bo')
# plt.plot(model.means[:,0],model.means[:,1],'ro')

# plt.show()
