import numpy as np


def loadDataSet():
    trainList = [[1,"s"], [1,"m"], [1,"m"], [1,"s"], [1,"s"],
                 [2,"s"], [2,"m"], [2,"m"], [2,"l"], [2,"l"],
                 [3,"l"], [3,"m"], [3,"m"], [3,"l"], [3,"l"]]
    trainCategory = [0,0,1,1,0,
                     0,0,1,1,1,
                     1,1,1,1,0]  # 1代表侮辱性文字，0代表正常言论
    return trainList, trainCategory

def createVocabList(dataset):
    vocabSet = set([])
    for word in dataset:
        vocabSet = vocabSet | set(word)
    return list(vocabSet)
    
def words2Vec(vocabList, inputDoc):#把一条记录转化为词条向量
    returnVec = [0] * len(vocabList)
    for word in inputDoc:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else :print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def createTrainMatrix(testDoc):
    trainList, trainCategory = loadDataSet()
    trainMatrix = []
    vocabList = createVocabList(trainList)
    for doc in trainList:
        trainMatrix.append(words2Vec(vocabList, doc))
    testDoc = words2Vec(vocabList, testDoc)
    return np.array(trainMatrix), np.array(trainCategory), np.array(testDoc)
    
        
def trainNB(trainMatrix, trainCategory):
    trainDocNum = len(trainMatrix)  #训练数据集大小
    wordsNum = len(trainMatrix[0])
    p1pro = sum(trainCategory) / float(trainDocNum) # 算出P（y = 1)概率
    p1Num = sum(trainCategory) # 分类为1的数量
    p1Num = 0 # 分类为1的所有单词的总数
    oneNum = np.zeros(wordsNum)
    p0Num = trainDocNum - p1Num # 分类为0的数量
    p0Num = 0 # 分类为0的所有单词的总数
    zeroNum = np.zeros(wordsNum)
    #计算每一个词在每种分类下的概率
    for i in range(trainDocNum):
        if trainCategory[i] == 1:
            oneNum  += trainMatrix[i]
            p1Num += sum(trainMatrix[i])
        else:
            zeroNum += trainMatrix[i]
            p0Num += sum(trainMatrix[i])
    
    p1Vec = oneNum / p1Num
    p0Vec = zeroNum / p0Num
    return p1Vec, p0Vec, p1pro #分别： 分类为1 时各词的概率；分类为0 时各词的概率；分类为1的概率

def classifyNB(vec2classify, p1Vec, p0Vec, p1pro):
    p1 =p1pro * sum(p1Vec * vec2classify)
    p0 = (1 - p1pro) * sum(p0Vec * vec2classify)
    if p1 > p0:
        return 1
    else :
        return 0
    
    ###########################################################################################################
    ###########################################################################################################
    ###########################################################################################################
    
def classify(trainDataSet, trainLabels, inputData):
    Py = {} # 各个y值的先验概率
    for label in trainLabels:
        Py[label] = (trainLabels.count(label) + 1 )/ (float(len(trainLabels)) + 2)
    #Pxy = {} #计算各y值情况下，特征x = inputData中对应特征的数量
    p_max = 0
    p_label = 0
    for y in Py.keys():
        y_index = [i for i, label in enumerate(trainLabels) if label == y]
        p_xy = Py[y]

        
        for j in range(len(inputData)):
            #dataset = np.array(trainDataSet) #np.array时，由于numpy中的数组是用来存放同一类型的数据的，所以在这里，整型被转化为了字符型
            #x_index = [i for i, x in enumerate(dataset[:, j]) if x == str(inputData[j])]
            x_index = [i for i, x in enumerate(list(map(lambda x:x[j],trainDataSet))) if x == str(inputData[j])]#map(lambda x:x[j],trainDataSet) 取trainDataSet的第j列
            xy_count = len(set(y_index)&set(x_index))
            p_xy = p_xy * (xy_count + 1)/ (float(trainDataSet.count(y)) + 3)
        if p_xy > p_max:
            p_max = p_xy
            p_label = y
    
    return p_max, p_label


    
if __name__ == "__main__":
    testVec = [2, 's']
    trainMatrix, trainCategory, testVec= createTrainMatrix(testVec)
    p1Vec, p0Vec, p1pro = trainNB(trainMatrix, trainCategory)
    print(classifyNB(testVec, p1Vec, p0Vec, p1pro))
    trainDataSet = [[1, "s"], [1, "m"], [1, "m"], [1, "s"], [1, "s"],
                 [2, "s"], [2, "m"], [2, "m"], [2, "l"], [2, "l"],
                 [3, "l"], [3, "m"], [3, "m"], [3, "l"], [3, "l"]]
    trainLabels = [0, 0, 1, 1, 0,
                     0, 0, 1, 1, 1,
                     1, 1, 1, 1, 0]
    inputData = [2, 's']
    
    p, y = classify(trainDataSet, trainLabels, inputData)
    print(p, y)