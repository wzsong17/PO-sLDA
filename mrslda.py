#encoding:utf8
'''
Created on 2016-4-4

@author: swz
'''
from pyhdfs import HdfsClient
from mrjob.job import MRJob
import numpy as np
from scipy.special import psi,gammaln
import os
import math


class VEM(MRJob):
    
    def mapper(self, _, line):
        try:
            linelist=line.split()
            docindex=int(linelist[0])
            label=int(linelist[1])
    
            iternum=int(os.environ['iternum'])
            fold=int(os.environ['fold'])
            
            if docindex%fold ==iternum%fold:
                K=int(os.environ['K'])
                V=int(os.environ['V'])
                C=int(os.environ['C'])
                alpha=float(os.environ['alpha'])
                outputdir=os.environ['outputdir']
                wordids,wordcts=self.parse_content(linelist[3:])
                N=len(wordids)

                #init gamma phi
                gamma = 1*np.random.gamma(100., 1./100., K)
                phi = np.ones((len(wordids), K))/float(K)
                sstats = np.zeros((K,V))
                grad_mu = np.zeros((C,K))
                #load lambda mu
                lamda=self.load_matrix(outputdir+'/lamda-%d' % (iternum-1),shape=(K,V))
                mu=self.load_matrix(outputdir+'/mu-%d' % (iternum-1),shape=(C,K))

                Elogbeta = self.dirichlet_expectation(lamda)
                Elogbetad=Elogbeta[:,wordids]
                expElogbeta = np.exp(Elogbeta)
                expElogbetad = expElogbeta[:,wordids]
                expmu = np.exp((1.0/N)*mu)
                expmud = expmu[label, :]
                for itr in range(int(os.environ['inner_max_iter'])):
                    Elogtheta = self.dirichlet_expectation(gamma)
                    expElogtheta = np.exp(Elogtheta)
                    
                    (h_phiprod, h) = self.calculatesfaux(phi, expmu, wordcts)
                    
                    lastgamma = gamma
                    gamma = alpha +\
                     np.sum (phi.T * wordcts, axis = 1)
                    
                    phi = (expElogtheta * expElogbetad.T) * expmud / np.exp(h/h_phiprod)
                    phinorm = np.sum(phi, axis = 1) + 1e-100
                    phi = (phi / phinorm[:,np.newaxis]) +1e-100
                    
                    meanchange = np.mean(abs(gamma - lastgamma))
                    if (meanchange < 0.001):
                        break
                sstats[:,wordids]+=  phi.T * wordcts

                grad_mu = grad_mu + self.calgradmu(phi, expmu, wordcts, label,C,K)
                likelihood = self.cal_locallikelihood(phi, wordcts, Elogtheta, Elogbetad, N, label, h_phiprod,mu,alpha,gamma,K)
             
		yield -2,likelihood*fold
                self.increment_counter('stat', 'num_of_document', 1)

                for i in range(K):
                    for j in range(N):
                        yield (i,wordids[j]),sstats[:,wordids][i,j]
                for c in range(C):
                    for k in range(K):
                        yield (-1,c,k) , grad_mu[c,k] #-1  mark for grad mu
        except Exception ,e:
            with open('/home/hadoop/tt.txt','a') as f:
                f.write(str(e))

    def reducer(self, key, value):
        yield key, sum(value)
        
    def parse_content(self, content):
        ids=[]
        cts=[]
        for word_count in content:
            tmp=word_count.split(':')
            ids.append(int(tmp[0]))
	    cts.append(int(math.log(int(tmp[1]),2.0))+1)
        return ids,cts

    def load_matrix(self,filepath,shape=None):
        if os.environ['local']=='1' and os.path.exists(filepath):
            return np.loadtxt(filepath,dtype=np.float)
        else:
            hosts=os.environ['hosts']
            if len(hosts)==0:
                hosts='master'
            client=HdfsClient(hosts=hosts)
            if client.exists(filepath):
                return np.fromstring(client.open(filepath).read()).reshape(shape)
        return False
    
    def dirichlet_expectation(self,alpha):
        """
        For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
        """
        if (len(alpha.shape) == 1):
            return(psi(alpha) - psi(np.sum(alpha)))
        return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])
    
    def calculatesfaux(self, phi, expmu, cts):
        sf_aux = np.dot(expmu, phi.T)

        sf_aux_power = np.power(sf_aux, cts)

	sf_aux_power=np.nan_to_num(sf_aux_power);
        
        sf_aux_prod = np.prod(sf_aux_power, axis = 1) + 1e-100
        h_phiprod = np.sum(sf_aux_prod)
	h_phiprod=np.nan_to_num(h_phiprod);
        h = np.zeros((phi.shape))
	h = np.nan_to_num(h)
        temp = (sf_aux_prod[:,np.newaxis] / sf_aux)
	temp =np.nan_to_num(temp)
        
        for v in range(0, len(h)):
            hvc = temp[:,v][:,np.newaxis] * expmu * cts[v]
            hv = np.sum(hvc, axis = 0)
            h[v,:] = hv
      
        return (h_phiprod, h)
    
    def calgradmu(self, phi, expmu, cts, label,C,K):
        gra_mu = np.zeros(expmu.shape)
        nphi = (phi.T * cts).T
        avephi = np.average(nphi, axis = 0)
        gra_mu[label,:] = avephi
        N = float(np.sum(cts))

        sf_aux = np.dot(expmu, phi.T)
        sf_aux_power = np.power(sf_aux, cts)
	sf_aux_power=np.nan_to_num(sf_aux_power);

        sf_aux_prod = np.prod(sf_aux_power, axis = 1) +1e-100
        kappa_1 = 1.0 / np.sum(sf_aux_prod)

        sf_pra = np.zeros((C, K))
        
        temp = (sf_aux_prod[:,np.newaxis] / sf_aux)
        for c in range (0, C):
            temp1 = np.outer(temp[c,:], (1.0/N) * expmu[c,:])
            temp1 = temp1 * nphi
            sf_pra[c,:] = np.sum(temp1, axis = 0)
        
        sf_pra = sf_pra * (-1) * kappa_1
        gra_mu = gra_mu + sf_pra
        return gra_mu
    def cal_locallikelihood(self, phi, cts, Elogthetad, Elogbetad, N, label, h_phiprod,mu,alpha,gamma,K):
        nphi = (phi.T * cts).T
        Elogpz_qz = np.sum(nphi * (Elogthetad - np.log(phi)))
        Elogpw = np.sum(nphi * Elogbetad.T)
        Elogpy = np.dot((1/N) * mu[label,:], np.average(nphi, axis = 0)) \
        - np.log(h_phiprod)
        likelihood = Elogpz_qz + Elogpw + Elogpy

        likelihood += np.sum((alpha - gamma)*Elogthetad)
        likelihood += np.sum(gammaln(gamma) - gammaln(alpha))
        likelihood += np.sum(gammaln(alpha*K) - gammaln(np.sum(gamma)))
            
        return likelihood  

    
if __name__ == '__main__':
    K=25
    D=5881
    V = 21007 # vocabulary size
    C = 19 #number of classes 
    alpha = 1./K #dirichlet parameter
    eta = 1./K #dirichlet parameter
    fold = 50
    kappa = 0.9 # stepsize parameter
    tau1 = 10. # stepsize parameter
    jobconf={
             'K' : K,
             'V' : V,# vocabulary size
             'C' : C,
             'max_it' : 100 ,# inner loop max iterations
             'fold' : fold,
             'outputdir':'/user/hadoop/song/congressional-bills',#/home/hadoop/workspace/mr.slda/data/',
             'local' : 0,#run on hadoop or local,1 for local
             'hosts':'master',#hosts of hdfs ,
             'alpha':alpha,
             'eta':eta,
             'inner_max_iter':100,
             }
    jobconf['iternum']=1
    VEM.JOBCONF=jobconf
    VEM.run()
        



    
