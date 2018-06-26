#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from Parameters import GetParameters, WriteParameters, MakeOutputFolders
from Loading import LoadImages, LoadModel, LoadData
from Training import TrainNetwork, TrainStatistics
from Testing import Classify, ClassStatistics

import time
StartTime = str(time.strftime("%d/%m/%Y"))+' '+str(time.strftime("%H:%M:%S"))

DataType, COSMOS, NTrainImages, NValidationImages, NTestImages, NetworkName, NumberOfEpochs, Threshold = GetParameters()

Directory = MakeOutputFolders(NetworkName)

WriteParameters(StartTime,Directory,DataType,COSMOS,NTrainImages,NValidationImages,NTestImages,NumberOfEpochs,Threshold)

Output = LoadData(NTrainImages+NValidationImages+NTestImages,DataType,COSMOS)

Images = LoadImages(Output,DataType)

TrainData = Output[:NTrainImages,:]
ValidationData = Output[NTrainImages:NTrainImages+NValidationImages,:]
TestData = Output[NTrainImages+NValidationImages:NTrainImages+NValidationImages+NTestImages,:]
TrainImages = Images[:NTrainImages,:,:,:]
ValidationImages = Images[NTrainImages:NTrainImages+NValidationImages,:,:,:]
TestImages = Images[NTrainImages+NValidationImages:NTrainImages+NValidationImages+NTestImages,:,:,:]

Model = LoadModel(Directory,TrainImages[0,:,:,:])

BestEpoch, Weights, Output, Accuracy = TrainNetwork(Model,TrainImages,ValidationImages,TrainData[:,1],ValidationData[:,1],NumberOfEpochs,Directory)

#TrainStatistics()

Classifications = Classify(Weights,TestImages,Model)

ClassStatistics(Classifications,TestData,Threshold,Directory)


