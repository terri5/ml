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
'''

reload(kNN)
normMat,ranges,minVals=kNN.autoNorm(datingDataMat)
norMat
...
...
'''

def datingClassTest():
    hoRatio  = 0.10
    datingDataMat,datingLabels= file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],5)
        print 'the classsifierResult came back with: %d,the real answer is: %d '\
                %(classifierResult,datingLabels[i])
        if(classifierResult != datingLabels[i]) : errorCount +=1.0
    print "the total error rate is: %f "%(errorCount/float(numTestVecs))
'''
kNN.datingClassTest()
'''

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("psercentage of time spent palying video games ?"))
    ffMiles = float(raw_input("requent flier miles earned per year ?"))
    iceCream = float(raw_input("liters of ice cream consumed per year ?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person :",\
            resultList[classifierResult - 1]
'''
kNN.classifyPerson()
'''

def img2vector(filename):
    returnVect = zeros((1,1024))
    print returnVect
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readlines()
        if lineStr:
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

'''
testVector = kNN.img2vector('testDigits/0_13.txt')
'''

from os import listdir

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s' %fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with : %d,the real answer is: %d"\
                %(classifierResult,classNumStr)
        if(classifierResult != classNumStr): errorCount +=1.0
    print "\nthe total number of tests is: %d" % mTest
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" %(errorCount/float(mTest))


'''
修正后的分类版本
'''
def classify1(inputPoint,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]     #已知分类的数据集（训练集）的行数
    #先tile函数将输入点拓展成与训练集相同维数的矩阵，再计算欧氏距离
    diffMat = tile(inputPoint,(dataSetSize,1))-dataSet  #样本与训练集的差值矩阵
    sqDiffMat = diffMat ** 2                    #差值矩阵平方
    sqDistances = sqDiffMat.sum(axis=1)         #计算每一行上元素的和
    distances = sqDistances ** 0.5              #开方得到欧拉距离矩阵
    sortedDistIndicies = distances.argsort()    #按distances中元素进行升序排序后得到的对应下标的列表
    #选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[ sortedDistIndicies[i] ]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

#文本向量化 32x32 -> 1x1024
def img2vector1(filename):
    returnVect = []
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect

#从文件名中解析分类数字
def classnumCut(fileName):
    fileStr = fileName.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr

#构建训练集数据向量，及对应分类标签向量
def trainingDataSet():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))                          #m维向量的训练集
    for i in range(m):
        fileNameStr = trainingFileList[i]
        hwLabels.append(classnumCut(fileNameStr))
        trainingMat[i,:] = img2vector1('trainingDigits/%s' % fileNameStr)
    return hwLabels,trainingMat


#测试函数
def handwritingTest1():
    import time
    hwLabels,trainingMat = trainingDataSet()    #构建训练集
    testFileList = listdir('testDigits')        #获取测试集
    errorCount = 0.0                            #错误数
    mTest = len(testFileList)                   #测试集总样本数
    t1 = time.time()
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = classnumCut(fileNameStr)
        vectorUnderTest = img2vector1('testDigits/%s' % fileNameStr)
       #调用knn算法进行测试
        classifierResult = classify1(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of tests is: %d" % mTest               #输出测试总样本数
    print "the total number of errors is: %d" % errorCount           #输出测试错误样本数
    print "the total error rate is: %f" % (errorCount/float(mTest))  #输出错误率
    t2 = time.time()
    print "Cost time: %.2fmin, %.4fs."%((t2-t1)//60,(t2-t1)%60)      #测试耗时

if __name__ == "__main__":
    handwritingTest1()