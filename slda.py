import numpy as n
import time
from scipy.special import  psi

n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

class SLDA_test:
    def __init__(self, V, K, C, D, mu, _lambda, max_it, alpha):
        self.iterations = 0
        self._K = K
        self._V = V
        self._C = C
        self._D = D
        self._max_it = max_it
        self._alpha = alpha
        #self._eta = eta # dirichlet parameters
        self._lambda = _lambda
        self._mu = mu
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._likelihoods = list()
        self._scores = n.zeros((D, self._C))
        self._predictions = list()
        
    def do_e_step(self, wordids, wordcts ):
        likelihood = 0.0
        #batchD = len(wordids)
	
        gamma = 1*n.random.gamma(100., 1./100., (self._D, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)
        for d in range(0, self._D):
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d,:]
            Elogthetad = Elogtheta[d,:]
            expElogthetad = expElogtheta[d,:]
	    
            expElogbetad = self._expElogbeta[:,ids]
	    
            Elogbetad = self._Elogbeta[:,ids]
            phi = n.ones((len(ids), self._K))/float(self._K)
            
            for it in range(0, self._max_it):
                lastgamma = gammad
                gammad = self._alpha +\
                 n.sum (phi.T * cts, axis = 1)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                
                phi = (expElogthetad * expElogbetad.T)
                phinorm = n.sum(phi, axis = 1) + 1e-100
                phi = phi / phinorm[:,n.newaxis]
                #phi = (phi.T / phinorm).T
                #phi_old = phi
                #nphi = (phi.T * cts).T
                
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            
            gamma[d, :] = gammad
            likelihood = likelihood + self.cal_loacllieklihood(phi, cts, Elogthetad, Elogbetad)
            self._scores[d,:] = n.dot(self._mu, n.average(phi.T * cts, axis = 1))
            self._predictions.append(n.argmax(self._scores[d,:]))
        
    
    def cal_loacllieklihood(self, phi, cts, Elogthetad, Elogbetad):

        nphi = (phi.T * cts).T
        Elogpz_qz = n.sum(n.sum(nphi * (Elogthetad - n.log(phi))))
        Elogpw = n.sum(n.sum(nphi * Elogbetad.T))
        likelihood = Elogpz_qz + Elogpw
            
        return likelihood
    
    def saveresults(self, it):
        n.save("scores.txt", self._scores)
        f = open("./predictions_%d.txt"%it, "w")
        for d in range(0, self._D):
            f.write(str(self._predictions[d]))
            f.write("\n") 
        f.close()
        
    def accuracy(self, goldlabel):
        right = 0
        for d in range(0, self._D):
            if (self._predictions[d] == goldlabel[d]):
                right = right + 1
        accuracy = float(right) / float(self._D)
        return accuracy 

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
