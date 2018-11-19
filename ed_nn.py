import matplotlib.pyplot as plt
import numpy as np
import time

class EDNN():

    def __init__(self,D,n_hidden=1,n_iter=100,print_every=1,crit=1e-5,parameter_init=None):
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
    		z.append(x)
    	return z

    def learn_identity(self,x):
    	#precompensate bias
    	data_mean=np.mean(x,axis=0)
    	self.encoder_parameters[0][1]=-data_mean
    	self.encoder_parameters[1][1]=data_mean

    	start_time = time.time()
    	print_time=self.print_every
    	step=1e-4
    	old_loss=[1e10]
    	z=self.forward(x,encoder_only=True)
    	de=z[-1]-x#nd-tensor
    	new_loss=np.einsum('nd,nd',de,de)#actually devided by two
    	improvement=np.abs((old_loss-new_loss)/old_loss)
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
    		old_loss=new_loss
    		new_loss=np.einsum('nd,nd',de,de)#actually devided by two¨
    		if old_loss-new_loss<0:
    			step/=2
    			print('decrease stepsize at epoch: '+str(iter))
    		improvement=0.9*improvement+0.1*(old_loss-new_loss)/np.abs(old_loss)#running average
    		current_time=time.time()-start_time
    		if current_time>print_time:
    			print('Epoch: '+str(iter)+', Training time: '+str(int(current_time))+'s, loss: '+str(new_loss)+', last improvement: '+str(improvement))
    			print_time=current_time+self.print_every
    			# plt.plot(x[0:100,0], x[0:100,1], 'b*')
    			# plt.plot(z[-1][0:100,0], z[-1][0:100,1], 'r*')
    			# plt.show()
    		iter+=1
    	self.copy_encoder_parameter()
    	return new_loss

    def copy_encoder_parameter(self):
    	for h in range(0,len(self.encoder_parameters)):
    		self.decoder_parameters[h][0]=np.copy(self.encoder_parameters[h][0])
    		self.decoder_parameters[h][1]=np.copy(self.encoder_parameters[h][1])


    def backprop_encoder(self,x,z,de,step=1e-5):
    	#z is a list of all activations
    	de_dA2=np.einsum('nd,nD->dD',de,z[-2])
    	self.encoder_parameters[-1][0]-=step*de_dA2
    	de_db2=np.einsum('nd->d',de)
    	self.encoder_parameters[-1][1]-=step*de_db2
    	#here we aactually need a for loop over n_hidden
    	de_dz1=np.einsum('nk,ki->ni',de,self.encoder_parameters[-1][0])
    	dtanh=1-np.square(z[-2])
    	dz1_dA1=np.einsum('ni,nj->nij',dtanh,x)
    	self.encoder_parameters[-2][0]-=step*np.einsum('nij,ni->ij',dz1_dA1,de_dz1)
    	self.encoder_parameters[-2][1]-=step*np.einsum('ni,ni->i',dtanh,de_dz1)

    def backprop_decoder(self,x,z,de,step=1e-5,update_decoder_only=False):
    	#z is a list of all activations
    	de_dA4=np.einsum('nd,nD->dD',de,z[-2])
    	self.decoder_parameters[-1][0]-=step*de_dA4
    	de_db4=np.einsum('nd->d',de)

    	self.decoder_parameters[-1][1]-=step*de_db4
    	
    	#h_loop in the decoder comes here
    	de_dz3=np.einsum('nk,ki->ni',de,self.decoder_parameters[-1][0])
    	dtanh_3=1-np.square(z[-2])
    	dz3_dA3=np.einsum('ni,nj->nij',dtanh_3,z[-3])
    	self.decoder_parameters[-2][0]-=step*np.einsum('nij,ni->ij',dz3_dA3,de_dz3)    	
    	self.decoder_parameters[-2][1]-=step*np.einsum('ni,ni->i',dtanh_3,de_dz3)

    	if not update_decoder_only:
	    	dz3_dz2=np.einsum('nl,li->nli',dthan_3,self.decoder_parameters[-2][0])
	    	de_dz2=np.einsum('nli,nl->ni',dz3_dz2,de_dz_3)
	    	dz2_dA2=z[-4]#(ND-tensor),no activation at the latent encoder ouput
	    	self.encoder_parameters[-1][0]-=step*np.einsum('nj,ni->ij',dz2_dA2,de_dz2)
	    	self.encoder_parameters[-1][1]-=step*np.einsum('ni->i',de_dz2)

	    	dz2_dz1=self.decoder_parameters[-3][0]#no activation at encoder output
	    	de_dz1=np.einsum('mi,nm->ni',dz2_dz1,de_dz_2)
	    	dtanh_1=1-np.square(z[-4])
	    	dz1_dA1=np.einsum('ni,nj->nij',z[-4],x)#(ND-tensor),no activation at the latent encoder ouput
	    	self.encoder_parameters[-2][0]-=step*np.einsum('nij,ni->ij',dz1_dA1,de_dz1)
	    	self.encoder_parameters[-2][1]-=step*np.einsum('ni,ni->i',dtanh_1,de_dz1)    	      	


















