import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.svm import SVC
import csv
from sklearn.decomposition import RandomizedPCA
from sklearn import preprocessing

#load data from CSV files
def loadDataSet(Path):
    data = []
    with open(Path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar = '|')
        for row in reader:
            featureList = row[0].split(',')
            for i in range(0, len(featureList)):
                featureList[i] = float(featureList[i])
            #print featureList
            data.append(featureList)
    return data

#feature selection with PCA
def pca2(Xtrain, Xtest):
    newTrain = []
    pca = RandomizedPCA(n_components=len(Xtrain[0])-30)
    pca.fit(Xtrain)
    newTrain = pca.transform(Xtrain)
    newTest = pca.transform(Xtest)
    return newTrain, newTest

#feature scaling
def scaleX(xVal):
    scaler = preprocessing.MinMaxScaler(feature_range= (0, 1))
    for i in range(0, len(xVal[0])):
        xVal[:, i] = scaler.fit_transform(np.float32(xVal[:, i]))
    
#generate testY
def getY():
    data = []
    for i in range(0,30):
        if(i < 15):
            data.append(0)
        else:
            data.append(1)
    return data

#svm classifier            
def svm(xVal, yVal, xTest):
    xVal = np.array(xVal)
    yVal = np.array(yVal)
    xTest = np.array(xTest)
    clf = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=1,
                gamma=0.2, kernel='rbf', max_iter=-1, probability=False,
                random_state=None, shrinking=True, tol=0.001, verbose=False)
    scaleX(xVal)
    scaleX(xTest)
    #xVal, xTest = pca2(xVal, xTest)
    #print xVal[0]
    
    #print xVal[0]
    
    #print yVal
    clf.fit(xVal, yVal)
    yTest2 = []
    ans = 0
    yTest = (clf.predict(xVal))
    print 'Classification on Traning Data'
    print yTest
    yTest2 = (clf.predict(xTest))
    print 'Classification on Test Data'
    print yTest2
    return yTest2
    

trainData = loadDataSet('training.csv')
testData = loadDataSet('test.csv')
ytrainData = getY()
ans = svm(trainData, ytrainData, testData)

