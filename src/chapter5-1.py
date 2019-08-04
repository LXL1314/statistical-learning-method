import numpy as np
import math

### 23行： RuntimeError: maximum recursion depth exceeded in comparison

def caculateEntropy(trainLabels):#输入当前集合的标签
    trainLabelsSet = set(trainLabels)
    H = 0
    trainNumber = len(trainLabels)
    for label in trainLabelsSet:
        pkey = trainLabels.count(label) / float(trainNumber)
        H  -= pkey*float(math.log(pkey,2))
    return H

def splitDataSet(trainDataSet, trainLabels, splitFeatureAxis, splitFeatureValue):
    trainData = np.array(trainDataSet)[...,splitFeatureAxis]
    #trainData = list(map(lambda x: x[splitFeatureAxis], trainDataSet))
    subDataSetCount = list(trainData).count(splitFeatureValue)
    lenDataSet = len(trainData)
    subDataSetLabels = []
    for i in range(lenDataSet):
        if trainData[i] == splitFeatureValue:
            subDataSetLabels.append(trainLabels[i])
    return subDataSetCount, subDataSetLabels

def chooseBestSplitFeatureAxis(trainDataSet,restFeatureAxisSet, trainLabels):#选择出最好的那个划分特征
    minGain = float('inf')
    bestFeatureAxis = 0
    for splitFeatureAxis in restFeatureAxisSet:
        gain = 0
        trainData = np.array(trainDataSet)[..., splitFeatureAxis]
        #trainData = list(map(lambda x: x[splitFeatureAxis], trainDataSet))
        splitFeatureValues = set(trainData)
        for splitFeatureValue in splitFeatureValues:
            subDataSetCount, subDataSetLabels = splitDataSet(trainDataSet, trainLabels, splitFeatureAxis, splitFeatureValue)
            gain += (subDataSetCount/float(len(trainDataSet)))*caculateEntropy(subDataSetLabels)
        if gain < minGain:
            minGain = gain
            bestFeatureAxis = splitFeatureAxis
    restFeatureAxisSet = [i for i in restFeatureAxisSet if i != bestFeatureAxis]
    restFeatureAxisSet = set(restFeatureAxisSet)
    return bestFeatureAxis, restFeatureAxisSet #选择完当前最好划分特征后，剩下的未被选择的划分特征

def findMaxLabel(trainLabels):
    labels = set(trainLabels)
    max = 0
    maxLabel = 0
    for label in labels:
        number = trainLabels.count(label)
        if number > max:
            max = number
            maxLabel = label
    return maxLabel

def createTree(trainDataSet,restFeatureAxisSet, trainLabels):
    if trainLabels.count(trainLabels[0]) == len(trainLabels):
        return trainLabels[0]
    elif len(restFeatureAxisSet) == 0:
        return findMaxLabel(trainLabels)
    
    bestFeatureAxis, restFeatureAxisSet = chooseBestSplitFeatureAxis(trainDataSet,restFeatureAxisSet, trainLabels)
    featureTrainData = np.array(trainDataSet)[..., bestFeatureAxis]
    #featureTrainData = list(map(lambda x: x[bestFeatureAxis], trainDataSet))
    featureValues = set(featureTrainData)
    tree = {bestFeatureAxis: {}}#键：分类特征，键值：该特征作为分为特征记录分类结果
    for featureValue in featureValues:
        subDataSet = [data for data in trainDataSet if data[bestFeatureAxis] == featureValue]
        subDataSetCount, subDataSetLabels = splitDataSet(trainDataSet, trainLabels, bestFeatureAxis, featureValue)
        tree[bestFeatureAxis][featureValue] = createTree(subDataSet, restFeatureAxisSet, subDataSetLabels)
    return tree

def classify_process(testData, tree):
    classifyFeatureAxis = list(tree.keys())[0]
    classifyResults = tree[classifyFeatureAxis]
    testDataLabel = 0
    for key in classifyResults.keys():
        if testData[classifyFeatureAxis] == key:
            if type(classifyResults[key]).__name__=='dict':
                testDataLabel = classify_process(testData, classifyResults[key])
            else:
                testDataLabel = classifyResults[key]
    return testDataLabel

def classify(testDataSet, tree):
    testDataLabels = []
    for testData in testDataSet:
        testDataLabel = classify_process(testData, tree)
        testDataLabels.append(testDataLabel)
    return testDataLabels

def cut(tree):
    pass

if __name__ == "__main__":
    trainDataSet = [["qingnian","no job","no house","yiban"], ["qingnian","no job","no house","hao"], ["qingnian","have job", "no house","hao"], ["qingnian","have job", "have house","yiban"], ["qingnian","no job","no house","yiban"],
                    ["zhongnian","no job","no house","yiban"], ["zhongnian","no job","no house","hao"], ["zhongnian","have job", "have house","hao"],["zhongnian","no job","have house","feichanghao"], ["zhongnian","no job","have house","feichanghao"],
                   ["laonian","no job", "have house","feichanghao"], ["laonian","no job","have house","hao"],  ["laonian","have job", "no house","hao"],  ["laonian","have job", "no house","feichanghao"],["laonian","no job","no house","yiban"]]
    restFeatureAxisSet = [i for i in range(len(trainDataSet[0]))]
    restFeatureAxisSet = set(restFeatureAxisSet)
    trainLabels = ["no", "no", "yes", "yes", "no",
                   "no", "no", "yes", "yes", "yes",
                   "yes", "yes", "yes", "yes", "no"]
    tree = createTree(trainDataSet,restFeatureAxisSet, trainLabels)
    print(tree)
    testDataSet = [["qingnian","have job", "no house", "feichanghao"]]
    testDataLabels = classify(testDataSet,tree)
    print(testDataLabels)
   
    