#encoding:utf8
'''
Created on 2016-4-5

@author: hadoop
'''
import os

def parse_ldac_to_mrslda():
    docfile='/home/hadoop/workspace/slda/images/train-data.dat'
    labfile='/home/hadoop/workspace/slda/images/train-label.dat'
    targetfile='/home/hadoop/workspace/slda/images/train1.txt'
    
    t=open(targetfile,'w')
    with open(docfile) as f1:
        with open(labfile) as f2:
            index=1
            for line in f1:
                lab=f2.readline().split()[0]
                t.write(str(index)+'\t'+lab+'\t'+line)
                index+=1
def clean_param_files():
    dirname='/home/hadoop/workspace/mr.slda/data/'
    top_idx=1
    remove_idx=0
    
    for i in range(10000):
        if os.path.exists(dirname+'mu-%d' % top_idx):
            if os.path.exists(dirname+'mu-%d' % remove_idx) and remove_idx%5 !=0:
                os.remove(dirname+'mu-%d' % remove_idx)
                print 'remove : ' +dirname+'mu-%d' % remove_idx
        if os.path.exists(dirname+'lamda-%d' % top_idx):
            if os.path.exists(dirname+'lamda-%d' % remove_idx) and remove_idx%5 !=0:
                os.remove(dirname+'lamda-%d' % remove_idx)
                print 'remove : ' +dirname+'lamda-%d' % remove_idx
        top_idx+=1
        remove_idx+=1
def parse_label_slda():
    with open('/home/hadoop/workspace/mr.slda/data/congressional-bills/testlab.txt') as s:
        with open('/home/hadoop/workspace/mr.slda/data/congressional-bills/slda_testlab.txt','w') as t:
            for line in s:
                ll=line.split()
                if len(ll)>1:
                    print >>t,ll[1]
#                     t.write(ll[1])
parse_ldac_to_mrslda()
        