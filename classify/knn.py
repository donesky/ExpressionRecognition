"""kNN: k Nearest Neighbors

Input:

#. newInput: vector to compare to existing dataset (1xN)

#. dataSet:  size m data set of known vectors (NxM)

#. labels: 	data set labels (1xM vector)

#. k: 		number of neighbors to use for comparison

Output:

#. the most popular class label



"""

from numpy import *
import operator

Emotion_dict= {1:'joie',2:'degout',3:'tristesse',4:'colere',5:'surprise'}
'''
Dictionary of emotions: In this project we use 5 base emotions.
'''
Matrix_dict={'joie':0,'degout':0,'tristesse':0,'colere':0,'surprise':0}
'''
Dictionary of result matrix: This dictionary is used for recording the results of each prediction.
'''


def createDataSet(dataList, lableList,lenTest):
    ''' Prepare data sets for predict

    :param dataList: The coordinates of all expressions in latent space.
    :param lableList: The classification labels of all expressions.
    :param lenTest: the number of the test data
    :return:
        groupData: Array list of the training data's coordinates

        groupTest: Array list of the testing data's coordinates

        labels: list of classification labels
    '''
    # create a matrix: each row as a sample
    groupData=array(dataList[0:len(dataList)-lenTest])
    groupTest=array(dataList[len(dataList)-lenTest:len(dataList)])
    lableList=lableList[0:len(dataList)-lenTest]
    labels = [Emotion_dict[x] if x in Emotion_dict else x for x in lableList]
    return groupData, groupTest, labels

def predictMatrix(dataList, lableList, lenTest, K,labelsAsListTest):
    '''Classify emotions using kNN then generate the predict matrix.

    :param dataList: The coordinates of all expressions in latent space.
    :param lableList: The classification labels of all expressions.
    :param lenTest: the number of the test data
    :param K: Number of neighbors to use for comparison
    :param labelsAsListTest: the real classification labels of testing datas.
    :return: Print the Matrix of results.
    '''
    groupData, groupTest, labels=createDataSet(dataList, lableList, lenTest)
    index=0
    for testPoint in groupTest:
        outputLabel,classCount = kNNClassify(testPoint, groupData, labels, K)
        for cC in classCount:
            classCount[cC]=str((classCount[cC]*100/K))+"%"
        print "Your input is:", Emotion_dict[labelsAsListTest[index]], "and classified to class: ", outputLabel
        print "Matrix of results is:",classCount,index
        print "-------------------------------------------------------------------"
        if Emotion_dict[labelsAsListTest[index]] == outputLabel:
            print '\033[94m'+" Correct!"+'\033[0m'
            Matrix_dict[outputLabel] += 1
        else:
            print '\033[91m'+"Wrong!"+'\033[0m'
        index += 1

    print "Results statistics:"
    print "------------------"
    print "|Emotion|   Accuracy|"
    countAll=0
    count = lenTest/Matrix_dict.__len__()
    for (k,v) in Matrix_dict.items():
        print str(k)+"  |"+str(v*100/count)+"%"
        countAll +=v
    print "------------------"
    print  "Total  |" + str(countAll * 100 / lenTest) + "%"

def predictImage(dataList, lableList, lenTest, K,labelsAsListTest):
    '''Classify emotions using kNN then generate the predict emotion.

    :param dataList: The coordinates of all expressions in latent space.
    :param lableList: The classification labels of all expressions.
    :param lenTest: the number of the test data
    :param K: Number of neighbors to use for comparison
    :param labelsAsListTest: the real classification labels of testing datas.
    :return: Print the Matrix of results.
    '''
    groupData, groupTest, labels=createDataSet(dataList, lableList, lenTest)
    index=0
    for testPoint in groupTest:
        outputLabel,classCount = kNNClassify(testPoint, groupData, labels, K)
        for cC in classCount:
            classCount[cC]=str((classCount[cC]*100/K))+"%"

    return outputLabel





# classify using kNN
def kNNClassify(newInput, dataSet, labels, k):
    ''' kNN method

    :param newInput: Data being classified
    :param dataSet: Training data
    :param labels: The classification labels
    :param k: nNumber of neighbors to use for comparison
    :return:
        maxIndex: The closest expression index.

        classCount: The matrix of the result.
    '''
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    ## step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distance)

    classCount = {'joie':0,'degout':0,'tristesse':0,'colere':0,'surprise':0}  # define a dictionary (can be append element)
    for i in xrange(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex,classCount


#Run the extraction work Independently.
if __name__ == '__main__':


    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']  # four samples and two classes
    #dataSet, labels = createDataSet()

    testX = array([1.2, 1.0])

    outputLabel,classCount = kNNClassify(testX, group, labels, 1)
    print "Your input is:", testX, "and classified to class: ", outputLabel

    testX = array([0.1, 0.3])
    outputLabel,classCount = kNNClassify(testX, group, labels, 3)
    print "Your input is:", testX, "and classified to class: ", outputLabel