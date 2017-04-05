"""This module is used to implement IO operations as a tool.

It is called after the face detection,the purpose is to save the detected feature values in a document for later use.
The operations for reading the detected feature values from the text are also implemented in this class.

.. note::

   Feature values and classification labels are stored in a different path.
    * Feature values are stored in ``FeaturesData/dataValues.txt``.
    * classification labels are stored in ``FeaturesData/labels.txt``.

"""

feature_values_path = '/Users/Alex/Documents/PRD/IHM_EmotionDetection/ProjetSI_FinalVersion/Ressources/Algorithmes/src/FeaturesData/dataValues.txt'
"""str: The path of the text which stored feature values.
Can be modified according to the requirements
"""

label_path = '/Users/Alex/Documents/PRD/IHM_EmotionDetection/ProjetSI_FinalVersion/Ressources/Algorithmes/src/FeaturesData/labels.txt'
"""str: The path of the text which stored labels of classification.
Can be modified according to the requirements
"""

def SaveFeatures(dataAsList):
    """ This function store detected feature values into text file .

    :param dataAsList: ({list}<type 'list'>) All the detected feature values.

    """

    fileDataValues = open(feature_values_path, 'w')
    for datas in dataAsList:
        fileDataValues.write(",".join(str(values) for values in datas))
        fileDataValues.write("\n")
    fileDataValues.close()

def ReadFeatures():
    """ This function read detected feature values from the text file .

    :return: A map(int, list) contains all the detected feature values.
    """

    fileDataValues = open(feature_values_path, 'r')
    yourResult = [line.split(',') for line in fileDataValues.readlines()]
    fileDataValues.close()
    return [map(int, resultline) for resultline in yourResult]

def SaveLabels(dataLabelsAsList):
    """This function store classification labels into text file.

    :param dataLabelsAsList:({list}<type 'int'>) All the classification labels.
    """

    fileDataValues = open(label_path, 'w')
    for datas in dataLabelsAsList:
        fileDataValues.write("%s\n" % datas)
    fileDataValues.close()

def ReadLabels():
    """ This function read classification labels from the text file .

    :return: A list(int) contains all the classification labels.
    """

    fileDataValues = open(label_path, 'r')
    yourResult = [line for line in fileDataValues.readlines()]
    fileDataValues.close()
    return [int(resultline) for resultline in yourResult]

if __name__ == '__main__':
    #1
    dataLabelsAsList = [1,2,3]
    SaveLabels(dataLabelsAsList)
    ReadLabels()
    #2
    dataAsList = {[1,2]}
    SaveFeatures(dataAsList)
    ReadFeatures()
