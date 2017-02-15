
"""
This module is responsible for saving features data to the text database.
"""
from extractFeaturesCoordinates import extractFeatures
from utilText import SaveFeatures, SaveLabels


def getListsFromImages(basepath_train):
    """ This function performs the function of face detection. And the results will be saved to two list.

    :param basepath_train: (str)Training data storage path.
    :return:
    dataAsList ({list}<type 'int'>):  A map contains all the detected feature values.

    labelsAsList list(int): A list contains all the classification labels.
    """

    # extracting features
    (res, facedb) = extractFeatures(basepath_train, False)
    if res == True:
        for key in facedb:
            print 'class', key
            i = 1
            for x in facedb[key]:
                print i, x
                i = i + 1
        print facedb
        # set results into lists
        dataAsList = []
        labelsAsList = []
        for key in facedb:
            for x in facedb[key]:
                array = []
                for y in x['points']:
                    array.append(y[0])
                    array.append(y[1])
                dataAsList.append(array)
                labelsAsList.append(key)
        return dataAsList,labelsAsList



#Run the extraction work Independently.
if __name__ == '__main__':
    #dataAsList, labelsAsList=getListsFromImages('ImagesTrain')
    dataAsList, labelsAsList=getListsFromImages('AdjustTrainData')

    SaveFeatures(dataAsList)
    SaveLabels(labelsAsList)




