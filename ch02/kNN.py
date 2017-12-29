# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 15:55
# @Author  : Terri
# @Site    : 
# @File    : kNN.py.py
# @Software: PyCharm
from numpy import *
import operator
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount= { }
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] =classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
'''
CMD:

import kNN
group,labels=kNN.createDataSet();
group
labels
group,labels=kNN.createDataSet();
'''


def file2matrix(filename):
     fr = open(filename)
     arrayOLines = fr.readlines()
     numberOfLines = len(arrayOLines)
     returnMat = zeros((numberOfLines,3))
     classLabelVector = []
     index = 0
     for line in arrayOLines:
         line=line.strip()
         listFromLine = line.split('\t')
         returnMat[index:] = listFromLine[0:3]
         classLabelVector.append(int(listFromLine[-1]))
         index += 1
     return returnMat,classLabelVector

'''
CMD:
reload(kNN)
datingDataMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')
datingDataMat
datingLabels[0:20]

#### ex A:
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax= fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
plt.show()

## ex B:
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
fig = plt.figure()
ax= fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15
.0*array(datingLabels))
plt.show()

'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals


