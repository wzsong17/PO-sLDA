import string
import numpy as n


def parse_docs(docs, doclabels):
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    wordids=list()
    wordcts=list()
    labels = list()
    for d in range(0,D):
        words=string.split(docs[d])
        doclabel = string.split(doclabels[d])
        label = int(doclabel[-1])
        labels.append(label)
        wordid=list()
        wordct=list()
        for i in range(1, len(words)):
            word=words[i].split(':')
            wordid.append(word[0])
            wordct.append(word[1])
        wordids.append(wordid)
        wordcts.append(wordct)

    for i in range(len(wordids)):
        for j in range(len(wordids[i])):
            wordids[i][j]=string.atoi(wordids[i][j])
            wordcts[i][j]=string.atoi(wordcts[i][j])



    return(wordids,wordcts, labels)


def parse_inputmodel(_lambda, mu, K, V, C):
    o_lambda = n.zeros((K, V))
    o_mu = n.zeros((C, K))
    for k in range (0, K):
        line = _lambda[k]
        ps = line.split()
        for v in range(0, V):
            o_lambda[k][v] = float(ps[v])
    for c in range(0, C):
        for k in range(0, K):
            line = mu[c]
            ps = line.split()
            for k in range(0, K):
                o_mu[c][k] = float(ps[k])
    
    return (o_lambda, o_mu)
    
