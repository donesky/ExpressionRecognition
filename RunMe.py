"""The main function of the project.

"""

from extractFeaturesCoordinates import extractFeatures
import getLatentSpace
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
import matplotlib.pyplot as plt
import dataSet.setFeaturesData as sfd
import classify.knn as knn
import GPy
import numpy as np
# get data from test images
#dataAsListTest, labelsAsListTest = sfd.getListsFromImages('ImagesTest')
dataAsListTest, labelsAsListTest = sfd.getListsFromImages('ImageData/InternetAdjust')

# create space latent by test data and train data
LatentModel, dataSamples, labelSamples = getLatentSpace.genLatentSpaceTest(dataAsListTest, labelsAsListTest,100, 2)
#LatentModel, dataSamples, labelSamples = getLatentSpace.genLatentSpace(100, 2)

print LatentModel
mu,var=LatentModel.predict(dataSamples)
print type(LatentModel)
#GPy.plotting.change_plotting_library('plotly')
#fig = LatentModel.plot_latent(labels=labelSamples)
#myplt = GPy.plotting.show(fig)
#plt.show(fig)
# GPy.models.GPLVM.
#a=LatentModel.__getattribute__('_predictive_variable')
#get predictive variables from LatentModel
numPoint=LatentModel.num_data
listPoint=[map(float, result[0:2]) for result in LatentModel._predictive_variable[0:numPoint]]
#get the predict Matrix using method KNN
knn.predictMatrix(listPoint,labelSamples,len(labelsAsListTest),6,labelsAsListTest)



