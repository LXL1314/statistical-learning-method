import numpy as np

def calculateGini(trainLabels):
    Gini = 1
    trainLabelsSet = set(trainLabels)
    trainNumber = len(trainLabels)
    for label in trainLabelsSet:
        Gini -= (trainLabels.count(label)/float(trainNumber))**2
    return Gini

def splitDataSet(trainDataSet, trainLabels, splitFeatureAxis, splitFeatureValue):
    splitFeatureData = np.array(trainDataSet)[...,splitFeatureAxis]
    subDataSet1 = []
    subDataSet1Labels = []
    subDataSet2 = []
    subDataSet2Labels = []
    for i in range(len(trainDataSet)):
        if splitFeatureData[i] == splitFeatureValue:
            subDataSet1.append(trainDataSet[i])
            subDataSet1Labels.append(trainLabels[i])
        else:
            subDataSet2.append(trainDataSet[i])
            subDataSet2Labels.append(trainLabels[i])
    return subDataSet1, subDataSet1Labels, subDataSet2, subDataSet2Labels
    
def chooseBestSplitFeatureValue(trainDataSet, trainLabels, splitFeatureAxis):
    minGini = float("inf")
    realSplitFeatureValue = 0
    splitFeatureData = np.array(trainDataSet)[..., splitFeatureAxis]
    splitFeatureValueSet = set(splitFeatureData)
    for splitFeatureValue in splitFeatureValueSet:
        subDataSet1, subDataSet1Labels, subDataSet2, subDataSet2Labels = splitDataSet(trainDataSet, trainLabels, splitFeatureAxis, splitFeatureValue)
        Gini = len(subDataSet1Labels)/float(len(trainDataSet))*calculateGini(subDataSet1Labels) + len(subDataSet2Labels)/float(len(trainDataSet))*calculateGini(subDataSet2Labels)
        if Gini <= minGini:
            minGini = Gini
            realSplitFeatureValue = splitFeatureValue
    return realSplitFeatureValue, minGini

def chooseBestSplitFeatureAxis(trainDataSet,restFeatureAxisSet, trainLabels):
    realMinGini = float("inf")
    bestSplitFeatureAxis = 0
    bestSplitFeatureValue = 0
    for featureAxis in restFeatureAxisSet:
        realSplitFeatureValue, minGini = chooseBestSplitFeatureValue(trainDataSet, trainLabels, featureAxis)
        if minGini < realMinGini:
            realMinGini = minGini
            bestSplitFeatureAxis = featureAxis
            bestSplitFeatureValue = realSplitFeatureValue
    restFeatureAxisSet = [i for i in restFeatureAxisSet if i != bestSplitFeatureAxis]
    restFeatureAxisSet = set(restFeatureAxisSet)
    return bestSplitFeatureAxis,bestSplitFeatureValue, restFeatureAxisSet

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
    
    bestSplitFeatureAxis, bestSplitFeatureValue, restFeatureAxisSet = chooseBestSplitFeatureAxis(trainDataSet,restFeatureAxisSet, trainLabels)
    tree = {bestSplitFeatureAxis:{bestSplitFeatureValue:{}}}
    subDataSet1, subDataSet1Labels, subDataSet2, subDataSet2Labels = splitDataSet(trainDataSet, trainLabels, bestSplitFeatureAxis, bestSplitFeatureValue)
    tree[bestSplitFeatureAxis][bestSplitFeatureValue]["="]  = createTree(subDataSet1, restFeatureAxisSet, subDataSet1Labels)
    tree[bestSplitFeatureAxis][bestSplitFeatureValue]["!="] = createTree(subDataSet2, restFeatureAxisSet, subDataSet2Labels)
    return tree

def classify_process(testData, tree):
    splitFeatureAxis = list(tree.keys())[0]
    splitFeatureValue = list(tree[splitFeatureAxis].keys())[0]
    splitResults = tree[splitFeatureAxis][splitFeatureValue]
    testDataLabel = 0
    for key in splitResults.keys():
        if testData[splitFeatureAxis] == splitFeatureValue:
            if type(splitResults["="]).__name__ == 'dict':
                testDataLabel = classify_process(testData, splitResults["="])
            else:
                testDataLabel = splitResults["="]
        else:
            if type(splitResults["!="]).__name__ == 'dict':
                testDataLabel = classify_process(testData, splitResults["!="])
            else:
                testDataLabel = splitResults["!="]
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