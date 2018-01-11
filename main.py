# encoding:utf8
'''
Created on 2016-4-4

@author: swz
'''
import mrslda
import time
import slda
import os
import numpy as np
from scipy.special import gammaln, psi
# from pyhdfs import HdfsClient
import parse_document
import readfiles
# import logging


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
	print "alpha.shape:"
	print alpha.shape
        return(psi(alpha) - psi(np.sum(alpha)))
    print "np.sum(alpha, 1):"
    print np.sum(alpha, 1)
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


def cal_globallikelihood(eta, Elogbeta, lamda, V):
    likelihood = 0.
    likelihood += np.sum((eta - lamda) * Elogbeta)
    print "eta:" 
    print eta
    print "Elogbeta:" 
    print Elogbeta
    print "lamda:"
    print lamda
    print "V:"
    print V
    likelihood += np.sum(gammaln(lamda) - gammaln(eta))
    likelihood += np.sum(gammaln(eta * V) - gammaln(np.sum(lamda, 1)))
    print "gammaln(lamda):"
    print gammaln(lamda)
    print "gammaln(eta):"
    print gammaln(eta)
    print "gammaln(eta * V):"
    print gammaln(eta * V)
    print "gammaln(np.sum(lamda, 1)):"
    print gammaln(np.sum(lamda, 1))

    return likelihood


def save_matrix(filepath, data, islocal=True, hosts='master'):
    if not os.path.exists(os.path.dirname(filepath)):
        os.mkdir(os.path.dirname(filepath))
    if islocal or islocal == '1':
        np.savetxt(filepath, data)
    else:
        client = HdfsClient(hosts=hosts)
        if client.exists(filepath):
            client.delete(filepath)

        client = HdfsClient(hosts=hosts)
        client.create(filepath, data.tostring())


def load_matrix(filepath, shape=None, islocal=True, hosts='master'):
    if islocal or islocal == '1':
        # np.savetxt(filepath,data)
        if os.path.exists(filepath):
            return np.loadtxt(filepath, dtype=np.float)
    else:
        client = HdfsClient(hosts=hosts)
        if client.exists(filepath):
            return np.fromstring(client.open(filepath).read()).reshape(shape)


def train():

    K = 10
    D = 93794
    V = 11727# 3401vocabulary size
    C = 10  # number of classes
    alpha = 1. / K  # dirichlet parameter
    eta = 1. / K  # dirichlet parameter
    fold = 400
    kappa = 0.9  # stepsize parameter
    tau1 = 10.  # stepsize parameter
    local = 1  # run on hadoop or local,1 for local
    rho3 = 0.08  # stepsize of  mu

    jobconf = {
        'K': K,
        'V': V,  # vocabulary size
        'C': C,
        'max_it': 100,  # inner loop max iterations
        'fold': fold,
        'outputdir': '/results',
        'local': local,
        'hosts': 'master',  # hosts of hdfs ,
        'alpha': alpha,
        'eta': eta,
        'inner_max_iter': 100,
        'mapreduce.job.maps': 30,
        'mapreduce.job.reduces': 10
    }
    if not os.path.exists(jobconf['outputdir']):
        os.mkdir(jobconf['outputdir'])
    # init param   lamda mu
    lamda = 1 * np.random.gamma(100., 1. / 100., (K, V))
    mu = 1 * np.random.gamma(100., 1. / 100., (C, K))  # softmax parameters
    
    logfile=open(jobconf['outputdir'] + '/train.log','w')
    
    save_matrix(jobconf['outputdir'] + '/lamda-0',
                lamda, islocal=jobconf['local'])
    save_matrix(jobconf['outputdir'] + '/mu-0', mu, islocal=jobconf['local'])

    for itr in range(1, 10000):

        starttime = time.time()

        jobconf['iternum'] = itr

#         mr_job = mrslda.VEM(args=['-r','hadoop','hdfs:///home/hadoop/song/mr.slda/data/LSHTC/dry_run/mrtrain.txt'])
        mr_job = mrslda.VEM(
            args=['-r', 'local', './traindoc.txt'])

        mr_job.JOBCONF = jobconf

        # run em
        with mr_job.make_runner() as runner:
            runner.run()
            gradlamda = np.zeros((K, V))
            gradmu = np.zeros((C, K))
            # read result from mapreduce job
            for line in runner.stream_output():
                key, value = mr_job.parse_output_line(line)
                if key == -2:  # -2 for sum of local_likelihood
                    local_likelihood = float(value)
                    print "key is -2"
                elif key[0] != -1:  # key[0] = -1 for grad of lamda
                    gradlamda[key[0], key[1]] = value
                else:  # grad of mu
                    gradmu[key[1], key[2]] = value
            try:
                num_doc = float(runner.counters()[0][
                                'stat']['num_of_document'])
            except:
                num_doc = 0  

            # update lamda
            lamda = load_matrix(jobconf['outputdir'] + '/lamda-%d' % (
                itr - 1), shape=(K, V), hosts='master', islocal=jobconf['local'])

            rho1 = np.power((itr + tau1), -kappa)
            gradlamda = -lamda + eta + gradlamda * D / float(num_doc)
            lamda = lamda + rho1 * gradlamda

            save_matrix(jobconf['outputdir'] + '/lamda-%d' %
                        itr, lamda, hosts='master', islocal=jobconf['local'])

            # update mu
            mu = load_matrix(jobconf['outputdir'] + '/mu-%d' % (itr - 1),
                             shape=(C, K), hosts='master', islocal=jobconf['local'])

            mu = mu + rho3 * gradmu

            save_matrix(jobconf['outputdir'] + '/mu-%d' %
                        itr, mu, hosts='master', islocal=jobconf['local'])
        #  free disk
        if os.path.exists(jobconf['outputdir'] + '/mu-%d' % (itr - 1)) and (itr - 1) % 5 != 0:
            os.remove(jobconf['outputdir'] + '/mu-%d' % (itr - 1))
        if os.path.exists(jobconf['outputdir'] + '/lamda-%d' % (itr - 1)) and (itr - 1) % 5 != 0:
            os.remove(jobconf['outputdir'] + '/lamda-%d' % (itr - 1))

        # calculate likelihood
        Elogbeta = dirichlet_expectation(lamda)
        global_likelihood = cal_globallikelihood(
            eta, Elogbeta, lamda, V) * fold
        likelihood = local_likelihood + global_likelihood
	print local_likelihood
	print global_likelihood

        costtime = time.time() - starttime
        print "iteration:%4.d, likelihood: %12.5f, num of doc : %d  costtime: %4.d s" % (itr, likelihood, num_doc, costtime)
        print >>logfile,"%4.d \t %12.5f \t %d \t %4.d" % (itr, likelihood, num_doc, costtime)
    logfile.close()


def test():

    docpath = "/testdoc.txt"
    labpath = "/testlab.txt"

    path = '/results'

    K = 10  # number of topics
    V = 11727  # vocabulary size
    C = 10  # number of classes
    alpha = 1. / K  # dirichlet parameter
    max_it = 100  # inner loop max iterations
    its = 10000
    accuracylist = list()
    maxa=0
    with open(path + "/accuracylist.txt", "w") as f:
        for it in range(1, its):
            lambdapath = path + "/lamda-%d" % (it * 5)
            mupath = path + "/mu-%d" % (it * 5)

            (docs, goldlabels, _lambda, mu) = \
                readfiles.readtestdata(docpath, labpath, lambdapath, mupath)

            (wordids, wordcts, goldlabels) = \
                parse_document.parse_docs(docs, goldlabels)
	    
            D = len(goldlabels)

            (i_lambda, i_mu) = parse_document.parse_inputmodel(_lambda, mu, K, V, C)

            SLDA = \
                slda.SLDA_test(V, K, C, D, i_mu, i_lambda, max_it, alpha)

            SLDA.do_e_step(wordids, wordcts)

            accuracy = SLDA.accuracy(goldlabels)
            if accuracy>maxa:
                maxa=accuracy
            print "iterations: %d, accuracy: %f" % (it, accuracy)
            print >>f, "%d\t%f" % (it, accuracy)
    print 'max accuracy is: %f' % maxa
if __name__ == '__main__':
    train()
  #test()
