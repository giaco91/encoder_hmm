import matplotlib.pyplot as plt
import numpy as np
import time

class EDNN():

    def __init__(self,D,n_hidden=1,n_iter=100,print_every=1,crit=1e-4,parameter_init=None):
        self.D=D
        self.n_hidden=n_hidden
        self.n_iter=n_iter
        self.print_every=print_every
        self.crit=crit
        if parameter_init is not None:
        	self.encoder_parameters=parameter_init[0]
        	self.decoder_parameters=parameter_init[1]

    def init_parameters(self,identity=False):
    	print('initialize_parameters..')
    	sigma=1
    	if identity==False:
    		self.encoder_parameters=[[sigma*np.random.randn(self.D,self.D),sigma*np.random.randn(self.D)]]
    		self.decoder_parameters=[[sigma*np.random.randn(self.D,self.D),sigma*np.random.randn(self.D)]]
	    	for h in range(0,self.n_hidden):
	    		self.encoder_parameters.append([sigma*np.random.randn(self.D,self.D),sigma*np.random.randn(self.D)])
	    		self.decoder_parameters.append([sigma*np.random.randn(self.D,self.D),sigma*np.random.randn(self.D)])
    	else:
    		self.encoder_parameters=[[np.eye(self.D),np.zeros(self.D)]]
    		self.decoder_parameters=[[np.eye(self.D),np.zeros(self.D)]]   		
	    	for h in range(0,self.n_hidden):
	    		self.encoder_parameters.append([np.eye(self.D),np.zeros(self.D)])
	    		self.decoder_parameters.append([np.eye(self.D),np.zeros(self.D)])

    def forward(self,x,encoder_only=False):
    	if len(x.shape)==1:
    		x=np.expand_dims(x, axis=0)
    	z=[]
    	x_hat=[]
    	for h in range(0,self.n_hidden):
    		x=np.einsum('dj,nj->nd',self.encoder_parameters[h][0],x)
    		x+=self.encoder_parameters[h][1]
    		x=np.tanh(x)
    		z.append(x)
    	x=np.einsum('dj,nj->nd',self.encoder_parameters[-1][0],x)
    	x+=self.encoder_parameters[-1][1]
    	z.append(x)
    	if encoder_only:
    		return z
    	else:
    		for h in range(0,self.n_hidden):
    			x=np.einsum('dj,nj->nd',self.decoder_parameters[h][0],x)
    			x+=self.decoder_parameters[h][1]
    			x=np.tanh(x)
    			z.append(x)
    		x=np.einsum('dj,nj->nd',self.decoder_parameters[-1][0],x)
    		x+=self.decoder_parameters[-1][1]
    		x_hat.append(x)
    	return z,x_hat

    def learn_identity(self,x):
    	#precompensate bias
    	data_mean=np.mean(x)
    	self.encoder_parameters[0][1]=-data_mean
    	self.encoder_parameters[1][1]=data_mean

    	start_time = time.time()
    	print_time=self.print_every
    	step=1e-4
    	loss=[1e10]
    	z=self.forward(x,encoder_only=True)
    	de=z[-1]-x#nd-tensor
    	loss.append(np.einsum('nd,nd',de,de))#actually devided by two
    	improvement=np.abs((loss[-2]-loss[-1])/loss[-2])
    	iter=0
    	# plt.plot(x[0:100,0], x[0:100,1], 'b*')
    	# plt.plot(z[-1][0:100,0], z[-1][0:100,1], 'r*')
    	# plt.show()
    	while improvement>self.crit:
    		#gradient descent step
    		self.backprop_encoder(x,z,de,step=step)
    		#forward evaluation
    		z=self.forward(x,encoder_only=True)
    		de=z[-1]-x#nd-tensor
    		loss.append(np.einsum('nd,nd',de,de))#actually devided by two¨
    		if loss[-2]-loss[-1]<0:
    			step/=2
    			print('decrease stepsize at epoch: '+str(iter))
    		improvement=0.9*improvement+0.1*(loss[-2]-loss[-1])/np.abs(loss[-2])
    		current_time=time.time()-start_time
    		if current_time>print_time:
    			print('Epoch: '+str(iter)+', Training time: '+str(int(current_time))+'s, loss: '+str(loss[-1])+', last improvement: '+str(improvement))
    			print_time=current_time+self.print_every
    			# plt.plot(x[0:100,0], x[0:100,1], 'b*')
    			# plt.plot(z[-1][0:100,0], z[-1][0:100,1], 'r*')
    			# plt.show()
    		iter+=1
    	return loss

    def backprop_encoder(self,x,z,de,step=1e-5):
    		#z is a list of all activations
    		de_dA2=np.einsum('nd,nD->dD',de,z[-2])
    		self.encoder_parameters[-1][0]-=step*de_dA2
    		#here we aactually need a for loop over n_hidden
    		de_dz1=np.einsum('nk,ki->ni',de,self.encoder_parameters[-1][0])
    		dtanh=1-np.square(z[-2])
    		dz1_dA1=np.einsum('ni,nj->nij',dtanh,x)
    		self.encoder_parameters[-2][0]-=step*np.einsum('nij,ni->ij',dz1_dA1,de_dz1)
    		de_db2=np.einsum('nd->d',de)
    		self.encoder_parameters[-1][1]-=step*de_db2
    		self.encoder_parameters[-2][1]-=step*np.einsum('ni,ni->i',dtanh,de_dz1)

   	def backprop_decoder(self,x,z,de,step=1e-5,update_bias=True)

















