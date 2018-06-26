#!/usr/bin/env python2
# -*- coding: utf-8 -*-

def GetParameters():
    
    # Ask for parameters for the CNN
    
    DataType = raw_input('Which image type should be used, Euclid is single-band, KiDS has 4 bands\n(E)uclid or (K)ids:\n').upper()
    COSMOS = raw_input('Use COSMOS realistic images only (Y/N)?:\n').upper()
    ImageMax = [100000,68923,60144]
    if COSMOS == 'N':
        ImageMax = str(ImageMax[0])
    elif DataType == 'E':
        ImageMax = str(ImageMax[1])
    else:
        ImageMax = str(ImageMax[2])
    TrainImages = raw_input('Enter the number of images to train the network with (Images available = {0}):\n'.format(ImageMax))
    if TrainImages == '':
        TrainImages = 10000
    else:
        TrainImages = int(TrainImages)
    ValidationImages = 500
    TestImages = raw_input('Enter the number of images to test on the trained network:\n')
    if TestImages == '':
        TestImages = 1000
    else:
        TestImages = int(TestImages)
    NetworkName = raw_input('Enter a name for your network, if blank the network will be named numerically:\n')
    NumberOfEpochs = int(raw_input('Enter the number of epochs to train for. Enter 0 to train the network until overfitting begins:\n'))
    Threshold = raw_input('Choose threshold over which images will be classified as lenses, between 0 and 1 (def = 0.5):\n')
    if Threshold == '':
        Threshold = 0.5
    else:
        Threshold = float(Threshold)

    return DataType, COSMOS, TrainImages, ValidationImages, TestImages, NetworkName, NumberOfEpochs, Threshold

def WriteParameters(StartTime,Directory,DataType,COSMOS,TrainImages,ValidationImages,TestImages,NumberOfEpochs,Threshold):
    
    # Writes the network parameters to a text file named NetworkParameters.txt
    
    print('\n\nWriting Network Parameters to file')
    
    file = open('Output/'+Directory+'/NetworkParameters.txt','w')
    file.write('Program ran at {0}\n\n'.format(StartTime))
    file.write('Network name:\t\t\t\t{0}\n'.format(Directory))
    if DataType == 'E':
        file.write('Image type:\t\t\t\tEuclid Simulations')
    else:
        file.write('Image type:\t\t\t\tKiDS Simulations')
    if COSMOS == 'Y':
        file.write('COSMOS-like?:\t\t\t\tYes')
    else:
        file.write('COSMOS-like?:\t\t\t\tNo')
    file.write('Number of images trained on:\t\t{0}\n'.format(TrainImages))
    file.write('Number of images validated on:\t\t{0}\n'.format(ValidationImages))
    file.write('Number of images classified on:\t\t{0}\n'.format(TestImages))
    file.write('Number of epochs trained for:\t\t{0}\n'.format(NumberOfEpochs))
    file.write('Threshold set as:\t\t\t{0}\n'.format(Threshold))
    file.close()
    
    print('Network Parameters written to:\t/Output/'+str(Directory)+'/NetworkParameters.txt')
    
    return

def MakeOutputFolders(NetworkName):
    
    import os
    
    if not os.path.isdir('Output'):
        os.makedirs('Output')
    
    if len(NetworkName) < 1:
        Namingsystem = 1
        while os.path.isdir('Output/'+str(str(Namingsystem).zfill(3))):
            Namingsystem = Namingsystem + 1
        Directory = str(str(Namingsystem).zfill(3))
        os.makedirs('Output/'+str(Directory))
    else:
        if os.path.isdir('Output/'+str(NetworkName)):
            Namingsystem = 1
            while os.path.isdir('Output/'+str(NetworkName)+'_'+str(str(Namingsystem)).zfill(3)):
                Namingsystem = Namingsystem + 1
            Directory = str(NetworkName)+'_'+str(str(Namingsystem)).zfill(3)
            os.makedirs('Output/'+str(Directory))
        else:
            Directory = str(NetworkName)
            os.makedirs('Output/'+str(Directory))
            
    os.makedirs('Output/'+Directory+'/ConvolutionKernels')
    
    return Directory
    