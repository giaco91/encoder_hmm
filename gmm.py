import numpy as np
import time

class GMM():

    def __init__(self,n_mixture,weights=None,means=None,covars=None,n_iter=100,print_every=1,crit=1e-5,verbose=False):
        self.n_mixture=n_mixture
        self.n_iter=n_iter
        self.print_every=print_every
        self.crit=crit
        self.weights=weights
        self.means=means
        self.covars=covars
        self.verbose=verbose

    def check_symmetric(self,A):
        symmetric=np.allclose(A, A.T)
        return symmetric
    def check_pos_def(self,A):
        posdef=np.all(np.linalg.eigvals(A) > 1e-4)#not too small eigenvalues
        return posdef

    def k_mean(self,data,K,rep=1,n_iter=1000,crit=1e-5,centroids=None):
        #data is an (N,D) numpy array
        #returns a list of centroids (np.array), clusters (list of np arrays), summed distance
        N=data.shape[0]#amount of data
        D=data.shape[1]#dimension of a data point

        best_model=[]
        #-----run k-means rep times
        for r in range(0,rep):
        
            #----init centroids:----
            centroids=np.zeros((K,D))#allocate memory
            mean=np.mean(data,axis=0)
            sigma=np.std(data,axis=0)
            #initialize centroids
            for d in range(0,D):
                centroids[:,d]=np.random.normal(mean[d],sigma[d],K)

            #-----k-mean iterations----
            if self.verbose:
                print('start k-mean repetition: '+str(r+1)+'...')
            clusters = [[] for _ in range(K)]
            #E-step: assign points to neaerest centroid
            last_distance=1e10
            new_distance=1e9
            best_model=[centroids,clusters,last_distance]
            iter=0
            while (last_distance-new_distance)/last_distance>crit and iter<n_iter:
                last_distance=new_distance
                new_distance=0
                clusters = [[] for _ in range(K)]
                for n in range(0,N):
                    squared_distance=np.sum(np.power(data[n,:]-centroids[:,:],2),axis=1)#broadcasting
                    #squared_distance = numpy.linalg.norm(data[n,:]-centroids[:,:])
                    idx_min=np.argmin(squared_distance)
                    clusters[idx_min].append(data[n,:].tolist())
                    new_distance+=np.power(squared_distance[idx_min],1/2)

                #M-step: move centroids to cluster center
                for k in range(0,K):
                    if clusters[k]:
                        centroids[k,:]=np.mean(clusters[k],axis=0)
                iter+=1
                if np.mod(iter,5)==0 and self.verbose:
                    print('iter: '+str(iter))
            if new_distance<best_model[2]:
                best_model=[centroids,clusters,new_distance]

        for k in range(0,K):
            best_model[1][k]=np.asarray(best_model[1][k])

        #check for small clusters
        for k in range(0,K):
            if best_model[1][k].shape[0]<2+int(N/K/10):
                #retrain completely
                if self.verbose:
                    cluster_size=best_model[1][k].shape[0]
                    print('Retrain k-mean because of a too small cluster ('+str(cluster_size)+').')
                best_model=self.k_mean(data,K,rep=rep,n_iter=n_iter)
        return best_model

    def init_params(self,x):
        N=x.shape[0]
        if self.means is None or self.weights is None or self.covars is None:
            if self.means is None:
                [centroids,clusters,sum_distance]=self.k_mean(x,self.n_mixture,rep=1,n_iter=20)
                self.means=centroids
            elif self.weights is None or self.covars is None:
                _,clusters,sum_distance=self.k_mean(x,self.n_mixture,rep=1,n_iter=20,centroids=self.means)
            self.weights=np.zeros(self.n_mixture)
            self.covars=[]
            self.invcovars=[]
            self.normalizations=[]
            for m in range(0,self.n_mixture):
                #self.means[m,:]=centroids[m]
                self.weights[m]=clusters[m].shape[0]/N
                #calculate variance of clusters
                variances_m=np.var(clusters[m],axis=0)
                covar_m=np.diag(variances_m)
                if self.check_symmetric(covar_m) and self.check_pos_def(covar_m):
                    self.covars.append(covar_m)
                    self.invcovars.append(np.linalg.inv(covar_m))
                    self.normalizations.append(np.power(2*np.pi,-self.D/2)*np.power(np.linalg.det(covar_m),-1/2))
                else:
                    raise ValueError('covars are not well initialized')
        else:
            if self.D!=len(self.means[0]):
                raise ValueError('The dimension of the data ('+str(self.D)+') does not consist with the initialized parameter dimension ('+str(len(self.means[0]))+')')
            if self.verbose:
                print('Using given initial parameters....')
        if np.abs(np.sum(self.weights)-1)>1e-8:
            raise ValueError('weights are not well initialized: '+str(self.weights)+' sum to: '+str(np.sum(self.weights)))

    def update_parameters(self,component,mean=None,covar=None):
        #---this check should be skipped for efficiency if used often and safely
        if mean is None:
            pass
        elif not mean.shape[0] == self.D:
            raise ValueError('The dimensions of the mean vector and the covariance matrix are not consistent.')
        else:
            self.mean[component,:]=mean
        if covar is None:
            pass
        elif not self.check_pos_def(covar):
            print('Warning: covariance matrix is not strictly positive definite. Reinitializing component...')
            #try to make it positive definite
            self.mean[component,:]=np.mean(self.means, axis=0)
            new_covar=np.zeros((self.D,self.D))
            for m in range(0,self.n_mixture):
                if m!=component:
                    new_covar+=self.covars[m]
            covar=new_covar/(self.n_mixture-1)

            if not self.check_pos_def(covar):
                raise ValueError('covar still not strictly positive definite')
            else:
                print('could make the covariance matrix positive definite')
                self.covars[component]=covar
                self.invcovars[component]=np.linalg.inv(covar)
                self.normalizations[component]=np.power(2*np.pi,-self.D/2)*np.power(np.linalg.det(self.covar),-1/2)
        elif not self.check_symmetric(covar):
            print('Warining: The covariance matrix is not symmetric! Make it symmetric...')
            covar=(covar+np.transpose(covar))/2

        else:
            self.covars[component]=covar
            self.invcovars[component]=np.linalg.inv(covar)
            self.normalizations[component]=np.power(2*np.pi,-self.D/2)*np.power(np.linalg.det(self.covar),-1/2)

    def fit(self,x,collapse=None,l=0.5):
        #x is a np array of shape (N,D), where N is the # of data points and D the dimension
        #collapse: the covariance matrix feels a force towards a small values if 
        #the squarroot of the determinant is smaller than collapse*2. The force points
        #towards its scaled version with sqrt of determinant equal to collapse
        #l describes the force
        self.D=x.shape[1]
        N=x.shape[0]
        self.init_params(x)

        #---EM-algorithm---
        start_time = time.time()
        print_time=self.print_every
        LL=[-1e10]
        LL.append(self.score(x))
        improvement=(LL[-1]-LL[-2])/np.abs(LL[-2])
        iter=0
        while improvement>self.crit and iter<self.n_iter:
            #----E-step:
            gamma=self.get_responsibilities(x)
            #----M-step:
            s_gamma=np.sum(gamma,axis=0)
            #---weights----
            self.weights=s_gamma/N
            #---means---
            self.means=np.einsum('nm,nd->md',gamma,x)
            s_gamma=np.expand_dims(s_gamma, axis=1)
            self.means=self.means/s_gamma#broad casting
            #---covars----
            for m in range(0,self.n_mixture):
                x_min_mu_m=x-self.means[m]
                empirical_covars=np.einsum('nd,nD->ndD',x_min_mu_m,x_min_mu_m)
                if improvement<self.crit*10 and collapse is not None:
                    collapse_crit=np.linalg.det(self.covars[m])**(1/2)
                    if collapse_crit<2*collapse:
                        new_covar_m=np.einsum('n,ndD->dD',gamma[:,m],empirical_covars)/s_gamma[m]
                        l=0.5#if l=1 nothing happens, if l=0, the determinant of the new covar becomes collapse*2
                        self.covars[m]=new_covar_m*(l+(1-l)*collapse**(2/self.D)/np.linalg.det(new_covar_m)**(1/self.D))
                        if self.verbose:
                            print('collapse activated at value '+str(collapse_crit)+' for component: '+str(m)+'with  mean: '+str(self.means[m]))
                            print('new collapse value: '+str(np.linalg.det(self.covars[m])**(1/2)))
                    
                else:
                    self.covars[m]=np.einsum('n,ndD->dD',gamma[:,m],empirical_covars)/s_gamma[m]

            LL.append(self.score(x))                
            improvement=(LL[-1]-LL[-2])/np.abs(LL[-2])
            current_time=time.time()-start_time
            if current_time>print_time and self.verbose:
                print('Epoch: '+str(iter)+', Training time: '+str(int(current_time))+'s, Likelihood: '+str(LL[iter+1])+', last improvement: '+str(improvement))
                print_time=current_time+self.print_every

            iter+=1

        return LL

    def get_responsibilities(self,x):
        #x is the full data set of shape (N,D)
        N=x.shape[0]
        gamma=np.zeros((N,self.n_mixture))
        for n in range(0,N):
            denominator=self.mixture_densitiy(x[n,:])
            for m in range(0,self.n_mixture):
                gamma[n,m]=self.weights[m]*self.component_density(m,x[n,:])
            gamma[n,:]=gamma[n,:]/denominator
        return gamma


    def sample(self,n,truncate=None,covar_bias=1):
        #truncate is an option to forbit sampling from the tails of the Gaussians.
        #truncate is the amount of standarddeviations we want to sample from
        #if truncate goes to infinity, this corresponds to truncate=None
        #covar_bias scales the covariance matrix for the sampling by the factor covar_bias**2
        #you must choose between truncate and covar_bias. If both are not None, its truncate over covar_bias
        sample=np.zeros((n,self.D))
        mixture_idx = np.linspace(0,self.n_mixture-1,self.n_mixture).astype(int)
        weights = self.weights
        components=np.random.choice(mixture_idx, n, p=weights)
        if truncate is not None:
            i=0
            if self.D==1:
                while i<n:
                    sample_point=np.random.normal(self.means[components[i]],np.sum(self.covars[components[i]]),1)
                    if (sample_point-self.means[components[i]])**2/np.sum(self.covars[components[i]])<=truncate**2:
                        sample[i,:]=sample_point
                        i+=1

            while i<n:
                sample_point=np.random.multivariate_normal(self.means[components[i]],self.covars[components[i]],1)
                if np.einsum('i,i',sample_point-self.means[components[i]],np.einsum('ij,j', self.invcovars[components[i]], sample_point-self.means[components[i]]))<=truncate**2:
                    sample[i,:]=sample_point
                    i+=1
        else:
            covars=self.covars
            for j in range(self.n_mixture):
                covars[j]*=covar_bias**2
            for i in range(0,n):
                sample[i,:]=np.random.multivariate_normal(self.means[components[i]],covars[components[i]],1)
        return sample

    def mixture_densitiy(self,x):
        if x.shape[0]!=self.D:
            raise ValueError('The evaluation point x has not dimension D!')
        p=0
        for m in range(0,self.n_mixture):
            p+=self.weights[m]*self.component_density(m,x)
        return p

    def score(self,x):
        LL=0
        for i in range(0,x.shape[0]):
            LL+=np.log(self.mixture_densitiy(x[i,:]))
        return LL
               

    def component_density(self,component,x):
        if x.shape[0]!=self.D:
            raise ValueError('The evaluation point x has not dimension D!')
        return self.normalizations[component]*np.exp(-(1/2)*np.einsum('i,i',x-self.means[component],np.einsum('ij,j', self.invcovars[component], x-self.means[component])))

