

def readtrainingdata(batchsize, docindex, docpath, labpath):
    docs = file(docpath).readlines()
    labels = file(labpath).readlines()
    
    if ((docindex + batchsize) <= len(docs)):
        outputdocs = docs[docindex: docindex + batchsize]
        outputlabs = labels[docindex: docindex + batchsize]
        docindex = docindex + batchsize + 1
    if ((docindex + batchsize) > len(docs)):
        outputdocs = docs[docindex:]
        outputlabs = labels[docindex:]
        docindex = batchsize - (len(docs) - docindex)
        outputdocs[0:0] = docs[0:docindex]
        outputlabs[0:0] = labels[0:docindex]
        docindex = docindex + 1
    
    return (outputdocs, outputlabs, docindex)

def readtrainingdata1(batchsize, docindex, docs, labels):
    
    if ((docindex + batchsize) <= len(docs)):
        outputdocs = docs[docindex: (docindex + batchsize)]
        outputlabs = labels[docindex: (docindex + batchsize)]
        docindex = docindex + batchsize + 1
        return (outputdocs, outputlabs, docindex)
    if ((docindex + batchsize) > len(docs)):
        outputdocs = docs[docindex:]
        outputlabs = labels[docindex:]
        docindex = batchsize - (len(docs) - docindex)
        outputdocs[0:0] = docs[0:docindex]
        outputlabs[0:0] = labels[0:docindex]
        docindex = docindex + 1
        return (outputdocs, outputlabs, docindex)
    
def readtestdata(docpath, labpath, lambdapath, mupath):
    docs = file(docpath).readlines()
    goldlabels = file(labpath).readlines()
    _lambda = file(lambdapath).readlines()
    mu = file(mupath).readlines()
    return (docs, goldlabels, _lambda, mu)
    
    
        